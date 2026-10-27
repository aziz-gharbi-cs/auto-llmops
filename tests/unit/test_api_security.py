"""Unit tests for API RBAC security rules and role hierarchy (Section 27)."""

import pytest
from fastapi import HTTPException

from llmops.interfaces.api.security import (
    AuthenticatedUser,
    Role,
    get_current_user,
    require_roles,
)


def test_role_hierarchy() -> None:
    admin = AuthenticatedUser(principal_id="admin", role=Role.ADMIN)
    assert admin.has_role(Role.ADMIN)
    assert admin.has_role(Role.ML_ENGINEER)
    assert admin.has_role(Role.DEVELOPER)
    assert admin.has_role(Role.VIEWER)

    dev = AuthenticatedUser(principal_id="dev", role=Role.DEVELOPER)
    assert not dev.has_role(Role.ADMIN)
    assert not dev.has_role(Role.ML_ENGINEER)
    assert dev.has_role(Role.DEVELOPER)
    assert dev.has_role(Role.VIEWER)

    viewer = AuthenticatedUser(principal_id="viewer", role=Role.VIEWER)
    assert not viewer.has_role(Role.DEVELOPER)
    assert viewer.has_role(Role.VIEWER)


def test_get_current_user_valid_api_key() -> None:
    user = get_current_user(api_key="dev-developer-key", bearer=None)
    assert user.role == Role.DEVELOPER
    assert user.has_role(Role.VIEWER)


def test_get_current_user_invalid_api_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(api_key="unknown-key", bearer=None)
    assert exc_info.value.status_code == 401


def test_require_roles_dependency() -> None:
    checker = require_roles([Role.ADMIN])
    admin_user = AuthenticatedUser(principal_id="admin", role=Role.ADMIN)
    viewer_user = AuthenticatedUser(principal_id="viewer", role=Role.VIEWER)

    # Allowed
    assert checker(admin_user) == admin_user

    # Forbidden
    with pytest.raises(HTTPException) as exc_info:
        checker(viewer_user)
    assert exc_info.value.status_code == 403
