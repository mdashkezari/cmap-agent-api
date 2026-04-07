#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- ECR settings ----
ECR_REPO="${ECR_REPO:-cmap-agent}"
IMAGE_LOCAL="${IMAGE_LOCAL:-cmap-agent:geo}"
IMAGE_TAG="${IMAGE_TAG:-geo}"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_URI="${ECR_REGISTRY}/${ECR_REPO}"
IMAGE_REMOTE="${ECR_URI}:${IMAGE_TAG}"

# ---- Platform settings ----
# ECS Fargate is often x86_64 by default, so linux/amd64 is the safest default on Apple Silicon.
# Single-arch: set DOCKER_PLATFORM=linux/amd64 or linux/arm64
# Multi-arch: set DOCKER_PLATFORMS="linux/amd64,linux/arm64" (uses buildx and pushes a manifest list)
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
DOCKER_PLATFORMS="${DOCKER_PLATFORMS:-$DOCKER_PLATFORM}"
BUILDER_NAME="${BUILDER_NAME:-cmap-agent-builder}"

# ---- S3 settings ----
S3_BUCKET="${S3_BUCKET:-cmap-agent-${AWS_ACCOUNT_ID}}"

echo "AWS_REGION=${AWS_REGION}"
echo "AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}"
echo "ECR image: ${IMAGE_LOCAL} -> ${IMAGE_REMOTE}"
echo "DOCKER_PLATFORMS=${DOCKER_PLATFORMS}"
echo "S3 bucket: ${S3_BUCKET}"

# --- Ensure S3 bucket exists ---
if aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
  echo "S3 bucket exists: ${S3_BUCKET}"
else
  echo "Creating S3 bucket: ${S3_BUCKET}"
  if [ "${AWS_REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}"
  else
    aws s3api create-bucket \
      --bucket "${S3_BUCKET}" \
      --region "${AWS_REGION}" \
      --create-bucket-configuration LocationConstraint="${AWS_REGION}"
  fi

  aws s3api put-public-access-block --bucket "${S3_BUCKET}" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

  aws s3api put-bucket-versioning --bucket "${S3_BUCKET}" \
    --versioning-configuration Status=Enabled
fi

# --- Ensure ECR repo exists ---
aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" 2>/dev/null || true

# --- Login to ECR ---
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# --- Build image (inject build metadata for /version) ---
# Set SKIP_BUILD=1 to skip building (if we already built ${IMAGE_LOCAL} for the desired platform).
SKIP_BUILD="${SKIP_BUILD:-0}"
BUILDX_PUSHED="0"

if [ "${SKIP_BUILD}" != "1" ]; then
  GIT_SHA="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  BUILD_TIME="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

  if [[ "${DOCKER_PLATFORMS}" == *","* ]]; then
    # Multi-arch build: must use buildx and push directly (cannot "docker push" a local multi-arch image).
    echo "Building + pushing multi-arch image to ${IMAGE_REMOTE} (GIT_SHA=${GIT_SHA}, BUILD_TIME=${BUILD_TIME})"

    if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
      docker buildx create --name "${BUILDER_NAME}" --use >/dev/null
    else
      docker buildx use "${BUILDER_NAME}" >/dev/null
    fi
    docker buildx inspect --bootstrap >/dev/null

    docker buildx build \
      --platform "${DOCKER_PLATFORMS}" \
      -f "${ROOT_DIR}/Dockerfile" \
      --build-arg GIT_SHA="${GIT_SHA}" \
      --build-arg BUILD_TIME="${BUILD_TIME}" \
      -t "${IMAGE_REMOTE}" \
      --push \
      "${ROOT_DIR}"

    BUILDX_PUSHED="1"
  else
    # Single-arch build: build locally, then tag+push.
    echo "Building ${IMAGE_LOCAL} for ${DOCKER_PLATFORMS} (GIT_SHA=${GIT_SHA}, BUILD_TIME=${BUILD_TIME})"
    docker build \
      --platform "${DOCKER_PLATFORMS}" \
      -f "${ROOT_DIR}/Dockerfile" \
      --build-arg GIT_SHA="${GIT_SHA}" \
      --build-arg BUILD_TIME="${BUILD_TIME}" \
      -t "${IMAGE_LOCAL}" \
      "${ROOT_DIR}"
  fi
else
  # If skipping build, multi-arch cannot be pushed unless it was built+pushed with buildx already.
  if [[ "${DOCKER_PLATFORMS}" == *","* ]]; then
    echo "ERROR: SKIP_BUILD=1 with multi-arch DOCKER_PLATFORMS requires a buildx --push build (no local multi-arch image to push)." >&2
    echo "Either set SKIP_BUILD=0 or set DOCKER_PLATFORMS to a single platform." >&2
    exit 2
  fi
fi

# --- Tag & push (single-arch path only) ---
if [ "${BUILDX_PUSHED}" != "1" ]; then
  docker tag "${IMAGE_LOCAL}" "${IMAGE_REMOTE}"
  docker image inspect "${IMAGE_REMOTE}" >/dev/null
  docker push "${IMAGE_REMOTE}"
fi

IMAGE_DIGEST="$(aws ecr describe-images \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPO}" \
  --image-ids imageTag="${IMAGE_TAG}" \
  --query 'imageDetails[0].imageDigest' \
  --output text 2>/dev/null || true)"

echo "Done."
echo "Pushed: ${IMAGE_REMOTE}"
if [ -n "${IMAGE_DIGEST}" ] && [ "${IMAGE_DIGEST}" != "None" ]; then
  echo "Digest: ${IMAGE_DIGEST}"
fi
echo "Artifacts bucket: ${S3_BUCKET}"
