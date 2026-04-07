#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"

# ---- Required ----
ECS_CLUSTER="${ECS_CLUSTER:-}"
ECS_SERVICE="${ECS_SERVICE:-}"

if [ -z "${ECS_CLUSTER}" ] || [ -z "${ECS_SERVICE}" ]; then
  echo "ERROR: ECS_CLUSTER and ECS_SERVICE are required." >&2
  echo "Example:" >&2
  echo "  ECS_CLUSTER=cmap-agent ECS_SERVICE=<service-name> IMAGE_TAG=geo-20260116013613 ./scripts/deploy_ecs.sh" >&2
  exit 2
fi

# ---- Optional ----
ECR_REPO="${ECR_REPO:-cmap-agent}"
IMAGE_TAG="${IMAGE_TAG:-geo}"
ECS_CONTAINER="${ECS_CONTAINER:-cmap-agent-container}"
ECS_WAIT_STABLE="${ECS_WAIT_STABLE:-1}"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_URI="${ECR_REGISTRY}/${ECR_REPO}"
NEW_IMAGE="${ECR_URI}:${IMAGE_TAG}"

echo "AWS_REGION=${AWS_REGION}"
echo "AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}"
echo "Deploying image: ${NEW_IMAGE}"
echo "ECS cluster: ${ECS_CLUSTER}"
echo "ECS service: ${ECS_SERVICE}"
echo "ECS container: ${ECS_CONTAINER}"

# Resolve current task definition ARN used by the service.
CUR_TD_ARN="$(aws ecs describe-services \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --query 'services[0].taskDefinition' \
  --output text)"

if [ -z "${CUR_TD_ARN}" ] || [ "${CUR_TD_ARN}" = "None" ]; then
  echo "ERROR: Could not resolve current task definition for service ${ECS_SERVICE}" >&2
  exit 3
fi

echo "Current task definition: ${CUR_TD_ARN}"

TMP_DIR="$(mktemp -d)"
TD_RAW_JSON="${TMP_DIR}/taskdef_raw.json"
TD_NEW_JSON="${TMP_DIR}/taskdef_new.json"

aws ecs describe-task-definition \
  --region "${AWS_REGION}" \
  --task-definition "${CUR_TD_ARN}" \
  --output json > "${TD_RAW_JSON}"

# Patch task definition JSON:
# - Remove read-only fields
# - Update the selected container image to NEW_IMAGE (tag-based)
python3 - "${TD_RAW_JSON}" "${TD_NEW_JSON}" "${ECS_CONTAINER}" "${NEW_IMAGE}" <<'PY'
import json
import sys

raw_path, out_path, container_name, new_image = sys.argv[1:5]

with open(raw_path, 'r', encoding='utf-8') as f:
    doc = json.load(f)

td = doc.get('taskDefinition') or {}

# Drop read-only / server-populated keys that register-task-definition will reject.
for k in [
    'taskDefinitionArn', 'revision', 'status', 'requiresAttributes',
    'compatibilities', 'registeredAt', 'registeredBy', 'deregisteredAt'
]:
    td.pop(k, None)

containers = td.get('containerDefinitions') or []
if not containers:
    raise SystemExit('No containerDefinitions found in task definition')

patched = False
for c in containers:
    if c.get('name') == container_name:
        c['image'] = new_image
        patched = True
        break

if not patched:
    # Fall back to first container if the requested name does not exist.
    containers[0]['image'] = new_image

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(td, f)
PY

# Register the new task definition revision.
NEW_TD_ARN="$(aws ecs register-task-definition \
  --region "${AWS_REGION}" \
  --cli-input-json "file://${TD_NEW_JSON}" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)"

if [ -z "${NEW_TD_ARN}" ] || [ "${NEW_TD_ARN}" = "None" ]; then
  echo "ERROR: Failed to register new task definition revision" >&2
  exit 4
fi

echo "Registered new task definition: ${NEW_TD_ARN}"

# Update service to use the new task definition.
aws ecs update-service \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${NEW_TD_ARN}" \
  >/dev/null

echo "Service update started: ${ECS_SERVICE} -> ${NEW_TD_ARN}"

if [ "${ECS_WAIT_STABLE}" = "1" ]; then
  echo "Waiting for service to become stable..."
  aws ecs wait services-stable \
    --region "${AWS_REGION}" \
    --cluster "${ECS_CLUSTER}" \
    --services "${ECS_SERVICE}"
  echo "Service is stable."
fi

echo "Done deploying ${NEW_IMAGE}"
