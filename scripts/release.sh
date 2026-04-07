#!/usr/bin/env bash
set -euo pipefail

# One-command release:
#   1) Build (optional) + push to ECR (tag-based)
#   2) Register a new ECS task definition revision pointing at that tag
#   3) Update the ECS service to the new revision
#
# Required env vars:
#   ECS_CLUSTER
#   ECS_SERVICE
#
# Optional env vars:
#   AWS_REGION (default: us-west-2)
#   ECR_REPO (default: cmap-agent)
#   IMAGE_TAG (default: geo-<UTC timestamp>)
#   ECS_CONTAINER (default: cmap-agent-container)
#   SKIP_BUILD=1 to skip docker build
#
# Example:
#   ECS_CLUSTER=cmap-agent \
#   ECS_SERVICE=cmap-agent-task-definition-service-hkywydl5 \
#   ./scripts/release.sh

AWS_REGION="${AWS_REGION:-us-west-2}"
ECS_CLUSTER="${ECS_CLUSTER:-}"
ECS_SERVICE="${ECS_SERVICE:-}"

if [ -z "${ECS_CLUSTER}" ] || [ -z "${ECS_SERVICE}" ]; then
  echo "ERROR: ECS_CLUSTER and ECS_SERVICE are required." >&2
  echo "Example:" >&2
  echo "  ECS_CLUSTER=cmap-agent ECS_SERVICE=<service-name> ./scripts/release.sh" >&2
  exit 2
fi

# Use an immutable default tag to avoid any 'same-tag' ambiguity.
if [ -z "${IMAGE_TAG:-}" ]; then
  export IMAGE_TAG="geo-$(date -u +%Y%m%d%H%M%S)"
fi

export AWS_REGION

echo "Using IMAGE_TAG=${IMAGE_TAG}"

"$(dirname "$0")/push_ecr.sh"
"$(dirname "$0")/deploy_ecs.sh"

echo "Release complete."
