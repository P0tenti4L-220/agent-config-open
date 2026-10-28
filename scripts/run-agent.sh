#!/bin/sh
# Daily launcher for the error triage agent.
#
# It loads only local credentials (local/deepseek.env, local/github.env), keeps
# acceptance fixtures (local/live-test.env) out of normal runs, and resolves
# caller-relative paths before changing into the repository. The Python CLI
# still owns configuration parsing, provider checks and run semantics.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
caller_pwd=$(pwd)
python_bin="$repo_root/.venv/bin/python"

if [ ! -x "$python_bin" ]; then
    cat >&2 <<EOF
未找到本项目解释器：$python_bin
请先安装依赖：
  cd $repo_root
  python3 -m venv .venv
  .venv/bin/python -m pip install -e '.[dev]'
EOF
    exit 2
fi

help_requested=0
for arg in "$@"; do
    case "$arg" in
        -h|--help|--version) help_requested=1 ;;
    esac
done
if [ "$help_requested" -eq 1 ]; then
    cd "$repo_root"
    exec "$python_bin" -m agent_triage "$@"
fi

# Help/version must work even when a local credential file is broken. For
# normal commands, local files supply defaults; exported credentials win.
caller_deepseek_key=${DEEPSEEK_API_KEY:-}
caller_github_token=${GITHUB_PERSONAL_ACCESS_TOKEN:-}
for env_file in "$repo_root/local/deepseek.env" "$repo_root/local/github.env"; do
    if [ -f "$env_file" ]; then
        set -a
        . "$env_file"
        set +a
    fi
done
if [ -n "$caller_deepseek_key" ]; then
    export DEEPSEEK_API_KEY="$caller_deepseek_key"
fi
if [ -n "$caller_github_token" ]; then
    export GITHUB_PERSONAL_ACCESS_TOKEN="$caller_github_token"
fi
unset caller_deepseek_key caller_github_token

command_name=
for arg in "$@"; do
    case "$arg" in
        -*) ;;
        *)
            command_name=$arg
            break
            ;;
    esac
done

resolve_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)
            case "$caller_pwd" in
                /) printf '/%s\n' "$1" ;;
                *) printf '%s/%s\n' "$caller_pwd" "$1" ;;
            esac
            ;;
    esac
}

quote_arg() {
    quoted=$(printf '%s' "$1" | sed "s/'/'\\\\''/g")
    printf "'%s'" "$quoted"
}

new_args=
append_arg() {
    new_args="$new_args $(quote_arg "$1")"
}

seen_config=0
seen_project_root=0
seen_output=0
seen_replay=0
config_path=

while [ "$#" -gt 0 ]; do
    arg=$1
    case "$arg" in
        --config|--project-root|--symptom-file|--output|--event|--replay|--cases|--evidence-dir|--bundle|--dataset|--subjects|--review|--outcome|--experiment-dir|--baseline|--candidate|--file)
            key=$arg
            if [ "$#" -lt 2 ]; then
                printf '缺少参数值：%s\n' "$arg" >&2
                exit 2
            fi
            value=$(resolve_path "$2")
            append_arg "$key"
            append_arg "$value"
            case "$key" in
                --config)
                    seen_config=1
                    config_path=$value
                    ;;
                --project-root) seen_project_root=1 ;;
                --output) seen_output=1 ;;
                --replay) seen_replay=1 ;;
            esac
            shift 2
            ;;
        --config=*|--project-root=*|--symptom-file=*|--output=*|--event=*|--replay=*|--cases=*|--evidence-dir=*|--bundle=*|--dataset=*|--subjects=*|--review=*|--outcome=*|--experiment-dir=*|--baseline=*|--candidate=*|--file=*)
            key=${arg%%=*}
            value=$(resolve_path "${arg#*=}")
            append_arg "$key=$value"
            case "$key" in
                --config)
                    seen_config=1
                    config_path=$value
                    ;;
                --project-root) seen_project_root=1 ;;
                --output) seen_output=1 ;;
                --replay) seen_replay=1 ;;
            esac
            shift
            ;;
        *)
            append_arg "$arg"
            shift
            ;;
    esac
done

if [ "$command_name" = "analyze" ]; then
    # A normal analyze uses the local DeepSeek configuration unless the caller
    # selected one or explicitly asked for replay. Replay keeps its own default.
    if [ "$seen_config" -eq 0 ] && [ "$seen_replay" -eq 0 ]; then
        default_config="$repo_root/local/config.toml"
        if [ ! -f "$default_config" ]; then
            cat >&2 <<EOF
未找到本机配置：$default_config
请先复制配置样板并填入本机值：
  mkdir -p "$repo_root/local"
  cp $repo_root/examples/deepseek.toml $default_config
DEEPSEEK_API_KEY 由 local/deepseek.env 或当前进程环境提供，不要写进 TOML。
EOF
            exit 2
        fi
        append_arg "--config"
        append_arg "$default_config"
        config_path=$default_config
        seen_config=1
    fi
    if [ "$seen_project_root" -eq 0 ]; then
        append_arg "--project-root"
        append_arg "$caller_pwd"
    fi
    if [ "$seen_output" -eq 0 ]; then
        append_arg "--output"
        append_arg "$repo_root/local/runs"
    fi
fi

eval "set -- $new_args"

if [ "$command_name" = "analyze" ] && [ -n "$config_path" ] && [ ! -f "$config_path" ]; then
    printf '配置文件不存在：%s\n' "$config_path" >&2
    exit 2
fi

# The CLI validates the effective provider's credential_env after --replay
# overrides the config. Do not impose a DeepSeek key on replay/other providers.
cd "$repo_root"
exec "$python_bin" -m agent_triage "$@"
