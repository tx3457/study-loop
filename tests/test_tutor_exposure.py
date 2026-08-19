"""Tutor Lab routes stay out of the default product surface."""

import unittest
from unittest.mock import patch

from fastapi import FastAPI

import main


class TestTutorExposure(unittest.TestCase):
    def test_default_main_application_does_not_advertise_tutor_lab(self):
        paths = main.app.openapi()["paths"]
        self.assertFalse(any(path.startswith("/agent/tutor") for path in paths))

    def test_tutor_routes_are_absent_when_lab_is_disabled(self):
        application = FastAPI()
        with patch.object(main, "supervisor_enabled", return_value=False):
            included = main._include_experimental_routers(application)

        self.assertFalse(included)
        paths = application.openapi()["paths"]
        self.assertFalse(any(path.startswith("/agent/tutor") for path in paths))

    def test_explicit_lab_enable_registers_tagged_tutor_routes(self):
        application = FastAPI()
        with patch.object(main, "supervisor_enabled", return_value=True):
            included = main._include_experimental_routers(application)

        self.assertTrue(included)
        schema = application.openapi()
        tutor_paths = {
            path: operations
            for path, operations in schema["paths"].items()
            if path.startswith("/agent/tutor")
        }
        self.assertEqual(
            set(tutor_paths),
            {
                "/agent/tutor/start",
                "/agent/tutor/submit",
                "/agent/tutor/oneshot",
                "/agent/tutor/assist",
                "/agent/tutor/assist/continue",
            },
        )
        for operations in tutor_paths.values():
            for operation in operations.values():
                self.assertIn("Experimental Tutor Lab", operation["tags"])


if __name__ == "__main__":
    unittest.main()
