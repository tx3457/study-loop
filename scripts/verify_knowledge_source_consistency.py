#!/usr/bin/env python3
"""Read-only check of graph/vector references in a dedicated knowledge database.

Run while the knowledge service is stopped or idle. Uses psql inside the given
database container; never changes data or calls model providers. A nonzero exit
means a derived source reference cannot resolve to an active canonical chunk.
"""

import argparse
import json
from pathlib import Path
import subprocess


def query(args, sql):
    result = subprocess.run(
        ["docker", "exec", args.container, "psql", "-U", args.user, "-d", args.database,
         "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout.strip())


def verify(args):
    tables = query(args, """
        SELECT COALESCE(json_agg(table_name ORDER BY table_name), '[]'::json)
        FROM information_schema.columns
        WHERE table_schema='public' AND column_name='chunk_ids'
          AND table_name LIKE 'lightrag_vdb_%'
    """)
    # Identifiers come only from the database catalog and are still quoted.
    vector_queries = [
        'SELECT workspace, unnest(chunk_ids) AS chunk_id FROM "'
        + table.replace('"', '""') + '"'
        for table in tables
    ]
    references = """
        SELECT workspace, regexp_split_to_table(properties->>'source_id', '<SEP>') AS chunk_id
        FROM lightrag_graph_nodes
        UNION ALL
        SELECT workspace, regexp_split_to_table(properties->>'source_id', '<SEP>') AS chunk_id
        FROM lightrag_graph_edges
    """
    if vector_queries:
        references += " UNION ALL " + " UNION ALL ".join(vector_queries)
    result = query(args, f"""
        WITH refs AS ({references}), invalid_refs AS (
            SELECT DISTINCT r.workspace, r.chunk_id
            FROM refs r
            LEFT JOIN lightrag_doc_chunks c ON c.workspace=r.workspace AND c.id=r.chunk_id
            LEFT JOIN sl_knowledge_bases k ON k.workspace=r.workspace
            LEFT JOIN sl_source_versions v ON v.id::text=c.full_doc_id AND v.knowledge_base_id=k.id
            LEFT JOIN sl_documents d ON d.id=v.document_id AND d.current_version_id=v.id
            WHERE COALESCE(r.chunk_id, '') != ''
              AND (c.id IS NULL OR v.status!='active' OR v.id IS NULL
                   OR d.id IS NULL OR d.status='deleted')
        )
        SELECT json_build_object(
            'reference_count', (SELECT count(*) FROM refs WHERE COALESCE(chunk_id, '')!=''),
            'invalid_reference_count', (SELECT count(*) FROM invalid_refs),
            'invalid_references', COALESCE((SELECT json_agg(s) FROM (
                SELECT * FROM invalid_refs ORDER BY workspace, chunk_id LIMIT 100
            ) s), '[]'::json)
        )
    """)
    result.update(
        status="pass" if result["invalid_reference_count"] == 0 else "failed_source_consistency",
        database=args.database, vector_tables=tables, read_only=True, model_calls=0,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", default="studyloop_graph")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
