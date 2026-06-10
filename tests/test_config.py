from minicode.config import merge_settings, validate_provider_runtime
from minicode.model_registry import Provider, build_provider_config, detect_provider


def test_merge_settings_merges_env_and_mcp_servers() -> None:
    merged = merge_settings(
        {
            "env": {"A": "1"},
            "mcpServers": {
                "fs": {"command": "npx", "args": ["a"], "env": {"X": "1"}}
            },
        },
        {
            "env": {"B": "2"},
            "mcpServers": {
                "fs": {"command": "uvx", "env": {"Y": "2"}},
                "search": {"command": "python"},
            },
        },
    )

    assert merged["env"] == {"A": "1", "B": "2"}
    assert merged["mcpServers"]["fs"]["command"] == "uvx"
    assert merged["mcpServers"]["fs"]["args"] == ["a"]
    assert merged["mcpServers"]["fs"]["env"] == {"X": "1", "Y": "2"}
    assert merged["mcpServers"]["search"]["command"] == "python"


def test_validate_provider_runtime_rejects_mismatched_provider_key() -> None:
    errors = validate_provider_runtime(
        {
            "model": "gpt-4o",
            "openaiApiKey": "",
            "apiKey": "anthropic-key-does-not-unlock-openai",
            "openaiBaseUrl": "https://api.openai.com",
        }
    )

    assert any("OPENAI_API_KEY" in error for error in errors)


def test_validate_provider_runtime_accepts_openrouter_prefixed_model() -> None:
    errors = validate_provider_runtime(
        {
            "model": "anthropic/claude-sonnet-4",
            "openrouterApiKey": "sk-or-test",
            "openrouterBaseUrl": "https://openrouter.ai/api",
        }
    )

    assert errors == []


def test_deepseek_provider_uses_openai_compatible_config(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    provider = detect_provider("deepseek-v4-flash", {})
    config = build_provider_config("deepseek-v4-flash", {})

    assert provider == Provider.DEEPSEEK
    assert config.is_openai_compatible is True
    assert config.base_url == "https://api.deepseek.com"
    assert config.api_key == "sk-deepseek-test"


def test_validate_provider_runtime_accepts_deepseek() -> None:
    errors = validate_provider_runtime(
        {
            "model": "deepseek-v4-flash",
            "deepseekApiKey": "sk-deepseek-test",
            "deepseekBaseUrl": "https://api.deepseek.com",
        }
    )

    assert errors == []
