#!/usr/bin/env bash
# Talk to the Lambda Cloud API for one single-GPU on-demand instance.
# Does not SSH, sync a dataset, or start training.
#
#   LAMBDA_CLOUD_API_KEY=... ./scripts/lambda_cloud.sh list-types
#   LAMBDA_CLOUD_API_KEY=... ./scripts/lambda_cloud.sh resolve
#   # --instance-type defaults to auto: cheapest pair, then closest to AWS us-west-1
#   LAMBDA_CLOUD_API_KEY=... ./scripts/lambda_cloud.sh launch \
#     --ssh-key-name <name>
#   LAMBDA_CLOUD_API_KEY=... ./scripts/lambda_cloud.sh status --instance-id <id>
#   LAMBDA_CLOUD_API_KEY=... ./scripts/lambda_cloud.sh terminate --instance-id <id>
#
# Operations: list-types, list-instances, list-ssh-keys, resolve, launch, status, terminate
#
# Launch refuses any instance type other than gpu_1x_*. The image is omitted so
# Lambda uses the default Lambda Stack image. The instance is left running.
#
# Environment:
#   LAMBDA_CLOUD_API_KEY   Lambda Cloud API key (Bearer token). Never printed.
#   GITHUB_RUN_ID    Stored on the instance as the github-run tag when set.

set -euo pipefail

API_BASE="https://cloud.lambda.com/api/v1"
POLL_INTERVAL_SECONDS=15
POLL_TIMEOUT_SECONDS=600

# Proximity to AWS us-west-1 (N. California). Lower is closer.
# Lambda us-west-1 is Lambda's California region, not the AWS region.
# A region missing from this table ranks 1000, then candidates sort by name.
REGION_RANKS='{
  "us-west-1": 0,
  "us-west-4": 1,
  "us-west-2": 2,
  "us-west-3": 3,
  "us-south-3": 4,
  "us-south-1": 5,
  "us-south-2": 6,
  "us-midwest-1": 7,
  "us-east-2": 8,
  "us-east-1": 9,
  "me-west-1": 10,
  "europe-central-1": 11,
  "asia-south-1": 12,
  "asia-northeast-2": 13,
  "asia-northeast-1": 14
}'

OPERATION=""
INSTANCE_TYPE="auto"
REGION=""
SSH_KEY_NAME=""
INSTANCE_ID=""
INSTANCE_NAME="lerobot-train"
HTTP_CODE=""
RESPONSE_BODY=""

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "[lambda_cloud] $*" >&2
}

require_api_key() {
    [[ -n "${LAMBDA_CLOUD_API_KEY:-}" ]] || die "LAMBDA_CLOUD_API_KEY is not set"
}

require_tools() {
    command -v curl >/dev/null || die "curl is required"
    command -v jq >/dev/null || die "jq is required"
}

require_single_gpu() {
    [[ "${INSTANCE_TYPE}" =~ ^gpu_1x_[a-z0-9_]+$ ]] || \
        die "instance type must be a single GPU (gpu_1x_*); got '${INSTANCE_TYPE}'"
}

require_instance_id() {
    [[ -n "${INSTANCE_ID}" ]] || die "--instance-id is required for ${OPERATION}"
    [[ "${INSTANCE_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid instance id"
}

# request METHOD PATH [JSON]
# Sets HTTP_CODE and RESPONSE_BODY. Must be called in the current shell.
request() {
    local method=$1
    local path=$2
    local data=${3:-}
    local tmp code
    local -a args

    tmp=$(mktemp)
    args=(
        -sS
        -o "${tmp}"
        -w '%{http_code}'
        --header 'accept: application/json'
        --header "Authorization: Bearer ${LAMBDA_CLOUD_API_KEY}"
        --request "${method}"
    )
    if [[ -n "${data}" ]]; then
        args+=(--header 'content-type: application/json' --data-binary "${data}")
    fi
    if ! code=$(curl "${args[@]}" "${API_BASE}${path}"); then
        rm -f "${tmp}"
        die "request failed: ${method} ${path}"
    fi
    HTTP_CODE=${code}
    RESPONSE_BODY=$(cat "${tmp}")
    rm -f "${tmp}"
}

require_2xx() {
    local method=$1
    local path=$2
    local body=$3
    if [[ "${HTTP_CODE}" -lt 200 || "${HTTP_CODE}" -ge 300 ]]; then
        echo "error: ${method} ${path} returned HTTP ${HTTP_CODE}" >&2
        echo "${body}" >&2
        exit 1
    fi
}

api() {
    local method=$1
    local path=$2
    local data=${3:-}
    request "${method}" "${path}" "${data}"
    require_2xx "${method}" "${path}" "${RESPONSE_BODY}"
    printf '%s' "${RESPONSE_BODY}"
}

print_instance() {
    local body=$1
    jq -r '
        .data
        | "instance_id=\(.id)",
          "status=\(.status)",
          "ip=\(.ip // "")",
          "name=\(.name // "")",
          "region=\(.region.name // "")",
          "instance_type=\(.instance_type.name // "")"
    ' <<<"${body}"
}

list_types() {
    local body
    body=$(api GET /instance-types)
    jq -r '
        .data
        | to_entries[]
        | [
            .key,
            .value.instance_type.description,
            "\(.value.instance_type.price_cents_per_hour) cents/hour",
            (
              [ .value.regions_with_capacity_available[]?.name ]
              | if length == 0 then "regions=-" else "regions=\(join(","))" end
            )
          ]
        | join("\t")
    ' <<<"${body}"
}

list_instances() {
    local body count
    body=$(api GET /instances)
    count=$(jq -r '.data | length' <<<"${body}")
    if [[ "${count}" == "0" ]]; then
        log "no running instances"
        return
    fi
    jq -r '
        .data[]
        | "instance_id=\(.id)\tname=\(.name // "")\tstatus=\(.status)\tip=\(.ip // "")\tinstance_type=\(.instance_type.name)\tregion=\(.region.name)"
    ' <<<"${body}"
}

list_ssh_keys() {
    local body count
    body=$(api GET /ssh-keys)
    count=$(jq -r '.data | length' <<<"${body}")
    if [[ "${count}" == "0" ]]; then
        log "no ssh keys"
        return
    fi
    jq -r '.data[].name' <<<"${body}"
}

# One TSV row per eligible pair, already sorted:
# type, region, price_cents_per_hour, rank, description.
# INSTANCE_TYPE=auto and an empty REGION consider every gpu_1x_* region with
# capacity. A set type or region narrows the rows. Sort is price, proximity
# to AWS us-west-1, type name, then region name.
ranked_candidates() {
    local body=$1
    jq -r \
        --arg type "${INSTANCE_TYPE}" \
        --arg region "${REGION}" \
        --argjson ranks "${REGION_RANKS}" '
        .data
        | to_entries
        | map(select(.key | test("^gpu_1x_[a-z0-9_]+$")))
        | map(select(if $type == "auto" or $type == "" then true else .key == $type end))
        | map(
            . as $entry
            | ($entry.value.regions_with_capacity_available // [])[]
            | select(if $region == "" then true else .name == $region end)
            | {
                type: $entry.key,
                region: .name,
                price: ($entry.value.instance_type.price_cents_per_hour | tonumber),
                rank: ($ranks[.name] // 1000),
                description: ($entry.value.instance_type.description // "")
              }
          )
        | sort_by(.price, .rank, .type, .region)
        | .[]
        | [.type, .region, (.price | tostring), (.rank | tostring), .description]
        | @tsv
    ' <<<"${body}"
}

die_no_candidate() {
    local body=$1
    if [[ "${INSTANCE_TYPE}" != "auto" ]]; then
        if [[ "$(jq -r --arg type "${INSTANCE_TYPE}" '.data[$type] == null' <<<"${body}")" == "true" ]]; then
            die "instance type not offered: ${INSTANCE_TYPE}"
        fi
        if [[ -n "${REGION}" ]]; then
            die "region ${REGION} has no capacity for ${INSTANCE_TYPE}"
        fi
        die "no capacity for ${INSTANCE_TYPE}"
    fi
    if [[ -n "${REGION}" ]]; then
        die "no single-GPU capacity in ${REGION}"
    fi
    die "no single-GPU instance type has capacity"
}

# When either side is unset, take the first ranked pair. Both set leaves the
# pair alone for load_instance_type.
resolve_instance_type() {
    local body price rank
    local -a lines
    [[ "${INSTANCE_TYPE}" == "auto" || -z "${REGION}" ]] || return 0
    body=$(api GET /instance-types)
    mapfile -t lines < <(ranked_candidates "${body}")
    if [[ ${#lines[@]} -eq 0 ]]; then
        die_no_candidate "${body}"
    fi
    IFS=$'\t' read -r INSTANCE_TYPE REGION price rank _ <<<"${lines[0]}"
    log "selected ${INSTANCE_TYPE} in ${REGION} at ${price} cents/hour rank=${rank}"
}

# Print the ranked pairs and the winner. Does not launch.
resolve_selection() {
    local body line type region price rank description
    local -a lines
    local n=0
    if [[ "${INSTANCE_TYPE}" != "auto" ]]; then
        require_single_gpu
    fi
    body=$(api GET /instance-types)
    mapfile -t lines < <(ranked_candidates "${body}")
    if [[ ${#lines[@]} -eq 0 ]]; then
        die_no_candidate "${body}"
    fi
    for line in "${lines[@]}"; do
        IFS=$'\t' read -r type region price rank description <<<"${line}"
        n=$((n + 1))
        printf 'candidate=%s\tinstance_type=%s\tregion=%s\tprice_cents_per_hour=%s\trank=%s\tdescription=%s\n' \
            "${n}" "${type}" "${region}" "${price}" "${rank}" "${description}"
    done
    IFS=$'\t' read -r type region price rank description <<<"${lines[0]}"
    printf 'instance_type=%s\nregion=%s\nprice_cents_per_hour=%s\nrank=%s\ndescription=%s\n' \
        "${type}" "${region}" "${price}" "${rank}" "${description}"
    log "resolved ${type} in ${region}; not launching"
}

load_instance_type() {
    local body entry regions price description rank
    body=$(api GET /instance-types)
    entry=$(jq -c --arg type "${INSTANCE_TYPE}" '.data[$type] // empty' <<<"${body}")
    [[ -n "${entry}" ]] || die "instance type not offered: ${INSTANCE_TYPE}"

    regions=$(jq -r '[ .regions_with_capacity_available[]?.name ] | join(",")' <<<"${entry}")
    price=$(jq -r '.instance_type.price_cents_per_hour' <<<"${entry}")
    description=$(jq -r '.instance_type.description' <<<"${entry}")

    if [[ -z "${regions}" ]]; then
        die "no capacity for ${INSTANCE_TYPE}"
    fi
    if [[ -z "${REGION}" ]]; then
        die "region was not selected for ${INSTANCE_TYPE}"
    elif [[ ",${regions}," != *",${REGION},"* ]]; then
        die "region ${REGION} has no capacity for ${INSTANCE_TYPE} (available: ${regions})"
    fi

    rank=$(jq -nr --argjson ranks "${REGION_RANKS}" --arg name "${REGION}" '$ranks[$name] // 1000')
    log "${INSTANCE_TYPE} \"${description}\" ${price} cents/hour region=${REGION} rank=${rank}"
}

launch_instance() {
    local payload body instance_id
    [[ -n "${SSH_KEY_NAME}" ]] || die "--ssh-key-name is required for launch"
    [[ ${#INSTANCE_NAME} -le 64 ]] || die "instance name must be 64 characters or fewer"
    require_api_key
    require_tools
    resolve_instance_type
    require_single_gpu
    load_instance_type

    payload=$(jq -n \
        --arg region "${REGION}" \
        --arg type "${INSTANCE_TYPE}" \
        --arg key "${SSH_KEY_NAME}" \
        --arg name "${INSTANCE_NAME}" \
        --arg run "${GITHUB_RUN_ID:-local}" \
        '{
            region_name: $region,
            instance_type_name: $type,
            ssh_key_names: [$key],
            name: $name,
            tags: [{key: "github-run", value: $run}]
        }')

    log "launching ${INSTANCE_TYPE} in ${REGION} as ${INSTANCE_NAME}"
    body=$(api POST /instance-operations/launch "${payload}")
    instance_id=$(jq -r '.data.instance_ids[0] // empty' <<<"${body}")
    [[ -n "${instance_id}" && $(jq -r '.data.instance_ids | length' <<<"${body}") == "1" ]] || \
        die "expected one instance id from launch"
    log "launched instance_id=${instance_id}"
    poll_until_ready "${instance_id}"
}

poll_until_ready() {
    local id=$1
    local deadline body status
    deadline=$(( $(date +%s) + POLL_TIMEOUT_SECONDS ))

    while true; do
        request GET "/instances/${id}"
        body=${RESPONSE_BODY}
        if [[ "${HTTP_CODE}" == "404" ]]; then
            log "instance ${id} not visible yet"
            status="booting"
        elif [[ "${HTTP_CODE}" -lt 200 || "${HTTP_CODE}" -ge 300 ]]; then
            echo "error: GET /instances/${id} returned HTTP ${HTTP_CODE}" >&2
            echo "${body}" >&2
            exit 1
        else
            status=$(jq -r '.data.status' <<<"${body}")
            log "instance ${id} status=${status} ip=$(jq -r '.data.ip // ""' <<<"${body}")"
        fi

        case "${status}" in
            active)
                print_instance "${body}"
                log "instance ${id} is active and was left running"
                return 0
                ;;
            unhealthy|terminated|terminating|preempted)
                if [[ "${HTTP_CODE}" != "404" ]]; then
                    print_instance "${body}"
                fi
                die "instance ${id} entered ${status}; terminate it to stop billing"
                ;;
        esac

        if (( $(date +%s) >= deadline )); then
            if [[ "${HTTP_CODE}" != "404" ]]; then
                print_instance "${body}"
            else
                echo "instance_id=${id}"
                echo "status=${status}"
                echo "ip="
            fi
            die "timed out after ${POLL_TIMEOUT_SECONDS}s waiting for ${id}; instance was left running"
        fi
        sleep "${POLL_INTERVAL_SECONDS}"
    done
}

status_instance() {
    local body
    require_instance_id
    body=$(api GET "/instances/${INSTANCE_ID}")
    print_instance "${body}"
}

terminate_instance() {
    local payload body
    require_instance_id
    payload=$(jq -n --arg id "${INSTANCE_ID}" '{instance_ids: [$id]}')
    log "terminating ${INSTANCE_ID}"
    body=$(api POST /instance-operations/terminate "${payload}")
    jq -r '
        .data.terminated_instances[]
        | "instance_id=\(.id)",
          "status=\(.status)",
          "ip=\(.ip // "")",
          "name=\(.name // "")"
    ' <<<"${body}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        list-types|list-instances|list-ssh-keys|resolve|launch|status|terminate)
            [[ -z "${OPERATION}" ]] || die "unexpected argument: $1"
            OPERATION=$1
            shift
            ;;
        --instance-type)
            [[ $# -ge 2 ]] || die "--instance-type requires a value"
            INSTANCE_TYPE=$2
            shift 2
            ;;
        --region)
            [[ $# -ge 2 ]] || die "--region requires a value"
            REGION=$2
            shift 2
            ;;
        --ssh-key-name)
            [[ $# -ge 2 ]] || die "--ssh-key-name requires a value"
            SSH_KEY_NAME=$2
            shift 2
            ;;
        --instance-id)
            [[ $# -ge 2 ]] || die "--instance-id requires a value"
            INSTANCE_ID=$2
            shift 2
            ;;
        --name)
            [[ $# -ge 2 ]] || die "--name requires a value"
            INSTANCE_NAME=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "${OPERATION}" ]] || die "operation is required (list-types, list-instances, list-ssh-keys, resolve, launch, status, terminate)"
if [[ "${OPERATION}" == "launch" || "${OPERATION}" == "resolve" ]] && [[ -z "${INSTANCE_TYPE}" ]]; then
    INSTANCE_TYPE="auto"
fi
[[ -n "${INSTANCE_NAME}" ]] || INSTANCE_NAME="lerobot-train"

case "${OPERATION}" in
    resolve)
        if [[ "${INSTANCE_TYPE}" != "auto" ]]; then
            require_single_gpu
        fi
        ;;
    launch)
        if [[ "${INSTANCE_TYPE}" != "auto" ]]; then
            require_single_gpu
        fi
        [[ -n "${SSH_KEY_NAME}" ]] || die "--ssh-key-name is required for launch"
        [[ ${#INSTANCE_NAME} -le 64 ]] || die "instance name must be 64 characters or fewer"
        ;;
    status|terminate)
        require_instance_id
        ;;
esac

require_api_key
require_tools

case "${OPERATION}" in
    list-types)
        list_types
        ;;
    list-instances)
        list_instances
        ;;
    list-ssh-keys)
        list_ssh_keys
        ;;
    resolve)
        resolve_selection
        ;;
    launch)
        launch_instance
        ;;
    status)
        status_instance
        ;;
    terminate)
        terminate_instance
        ;;
esac
