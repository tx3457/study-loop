from __future__ import annotations


class ServiceError(Exception):
    status_code = 400
    code = "bad_request"


class Conflict(ServiceError):
    status_code = 409
    code = "conflict"


class Expired(ServiceError):
    status_code = 410
    code = "expired"


class TooLarge(ServiceError):
    status_code = 413
    code = "too_large"


class Unavailable(ServiceError):
    status_code = 503
    code = "unavailable"
