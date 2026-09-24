/* Turn a /health/providers result into the few lines a first-time user needs.

   Several capabilities usually share one variable: a freshly copied
   .env.example makes chat, structured output and embeddings all report
   LLM_API_KEY. Grouping by variable says each fix once and lists who needs it. */

const CAPABILITY_LABELS = {
  chat: '对话模型',
  structured: '结构化输出',
  embedding: '向量模型',
}

const FIELD_LABELS = {
  api_key: 'API Key',
  base_url: '服务地址',
  model: '模型名',
}

const ISSUE_KIND_LABELS = {
  missing: '未填写',
  placeholder: '仍是示例占位值',
  invalid: '格式无效',
}

function describeFix(fix) {
  const issue = typeof fix?.issue === 'string' ? fix.issue : ''
  const cut = issue.lastIndexOf('_')
  const field = cut > 0 ? issue.slice(0, cut) : issue
  const kind = cut > 0 ? issue.slice(cut + 1) : ''
  const target = typeof fix?.env === 'string' && fix.env
    ? `.env 中的 ${fix.env}`
    : FIELD_LABELS[field] || '配置项'
  return {
    key: `fix:${fix?.env || issue}`,
    text: `${target} ${ISSUE_KIND_LABELS[kind] || '有误'}`,
  }
}

/** @returns {string[]} one line per distinct problem; empty when ready or unknown. */
export function describeProviderProblems(health) {
  if (!health || health.status === 'ready' || typeof health.providers !== 'object') return []
  const groups = new Map()
  const add = (key, text, capability) => {
    const entry = groups.get(key) || { text, capabilities: [] }
    if (!entry.capabilities.includes(capability)) entry.capabilities.push(capability)
    groups.set(key, entry)
  }

  for (const [capability, result] of Object.entries(health.providers || {})) {
    const label = CAPABILITY_LABELS[capability] || capability
    if (result?.status === 'misconfigured') {
      for (const fix of Array.isArray(result.fix_env) ? result.fix_env : []) {
        const { key, text } = describeFix(fix)
        add(key, text, label)
      }
    } else if (result?.status === 'unavailable') {
      add('unreachable', '无法连接模型服务，请检查服务地址、API Key 与网络', label)
    } else if (result?.code === 'model_not_listed') {
      add(
        `model:${result.model}`,
        `服务商的模型列表里没有 ${result.model}，请核对模型名`,
        label,
      )
    }
  }

  return [...groups.values()].map(
    ({ text, capabilities }) => `${text}（影响：${capabilities.join('、')}）`,
  )
}
