"""Authentication and role-based access control (RBAC) (Section 27/28)."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer


class Role(StrEnum):
    """User and service roles conforming to Section 27 of the specification."""

    VIEWER = "viewer"
    DEVELOPER = "developer"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"


# Role hierarchy mapping: higher roles inherit permissions of lower roles
ROLE_HIERARCHY: dict[Role, set[Role]] = {
    Role.VIEWER: {Role.VIEWER},
    Role.DEVELOPER: {Role.VIEWER, Role.DEVELOPER},
    Role.ML_ENGINEER: {Role.VIEWER, Role.DEVELOPER, Role.ML_ENGINEER},
    Role.ADMIN: {Role.VIEWER, Role.DEVELOPER, Role.ML_ENGINEER, Role.ADMIN},
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Configurable static keys for local development and service-to-service auth
DEV_API_KEYS: dict[str, Role] = {
    "dev-viewer-key": Role.VIEWER,
    "dev-developer-key": Role.DEVELOPER,
    "dev-mlops-key": Role.ML_ENGINEER,
    "dev-admin-key": Role.ADMIN,
}


class AuthenticatedUser:
    """Represents an authenticated principal with assigned roles."""

    def __init__(self, principal_id: str, role: Role) -> None:
        self.principal_id = principal_id
        self.role = role

    def has_role(self, required_role: Role) -> bool:
        """Verify if principal holds required role according to hierarchy."""
        return required_role in ROLE_HIERARCHY.get(self.role, set())


def get_current_user(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> AuthenticatedUser:
    """Validate credentials and return AuthenticatedUser principal."""
    # 1. API Key check
    if api_key:
        role = DEV_API_KEYS.get(api_key)
        if role:
            return AuthenticatedUser(principal_id=f"key-{api_key[:8]}", role=role)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # 2. Bearer token check
    if bearer:
        token = bearer.credentials
        if token == "mock-admin-token":
            return AuthenticatedUser(principal_id="user-admin", role=Role.ADMIN)
        if token == "mock-developer-token":
            return AuthenticatedUser(principal_id="user-developer", role=Role.DEVELOPER)
        # Default authenticated viewer
        return AuthenticatedUser(principal_id="user-jwt", role=Role.VIEWER)

    # In local development mode without auth headers, allow viewer access
    return AuthenticatedUser(principal_id="anonymous", role=Role.VIEWER)


def require_roles(allowed_roles: Sequence[Role]) -> Any:
    """Dependency factory enforcing RBAC on route endpoints."""

    def role_checker(
        current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    ) -> AuthenticatedUser:
        if any(current_user.has_role(r) for r in allowed_roles):
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Operation requires one of: {[r.value for r in allowed_roles]}",
        )

    return role_checker
