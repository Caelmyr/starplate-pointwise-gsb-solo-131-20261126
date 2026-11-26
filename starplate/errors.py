from dataclasses import dataclass


@dataclass
class RequestError(Exception):
    message: str
    field: str = ""
    status: int = 400

    def payload(self):
        problem = {"error": self.message}
        if self.field:
            problem["field"] = self.field
        return problem


def require(condition, message, field="", status=400):
    if not condition:
        raise RequestError(message, field, status)
