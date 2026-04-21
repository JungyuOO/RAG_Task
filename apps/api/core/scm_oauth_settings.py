from __future__ import annotations

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class ScmOauthSettings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    scm_github_client_id: str = ""
    scm_github_client_secret: str = ""
    scm_github_scope: str = "read:user repo"
    scm_github_authorize_url: str = "https://github.com/login/oauth/authorize"
    scm_github_token_url: str = "https://github.com/login/oauth/access_token"
    scm_github_user_url: str = "https://api.github.com/user"
    scm_github_host_url: str = "https://github.com"

    scm_gitlab_client_id: str = ""
    scm_gitlab_client_secret: str = ""
    scm_gitlab_scope: str = "read_user api"
    scm_gitlab_authorize_url: str = "https://gitlab.com/oauth/authorize"
    scm_gitlab_token_url: str = "https://gitlab.com/oauth/token"
    scm_gitlab_user_url: str = "https://gitlab.com/api/v4/user"
    scm_gitlab_host_url: str = "https://gitlab.com"


def get_scm_oauth_settings() -> ScmOauthSettings:
    return ScmOauthSettings()
