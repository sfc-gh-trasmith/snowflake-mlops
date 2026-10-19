#!/bin/bash
# Open a pull request that records the live gateway split in deploy/gateway-config.yml.
#
# deploy-prod, rollback, and traffic-shift change the split in Snowflake. main
# is protected, so the file can't be pushed directly; without a PR it goes
# stale, and the next push that touches it would apply an old split.
#
# Usage: bash scripts/sync_gateway_config.sh "<commit / PR title>"
# Needs: GH_TOKEN with contents:write and pull-requests:write.

set -euo pipefail

TITLE="$1"
if git diff --quiet -- deploy/gateway-config.yml; then
    echo "deploy/gateway-config.yml already matches the live split"
    exit 0
fi

BRANCH="gateway-sync/${GITHUB_RUN_ID:-local}"
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git checkout -b "$BRANCH"
git add deploy/gateway-config.yml
git commit -m "$TITLE"
git push -u origin "$BRANCH"

BODY="Records the gateway split that ${GITHUB_WORKFLOW:-a workflow} applied in Snowflake (run ${GITHUB_SERVER_URL:-}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-}).

Merge this so \`deploy/gateway-config.yml\` on \`main\` matches what is serving. Merging re-applies the same split (no traffic change) and runs the REST smoke test."
if ! gh pr create --base main --head "$BRANCH" --title "$TITLE" --body "$BODY"; then
    echo "::warning::Could not open a PR for deploy/gateway-config.yml. Enable 'Allow GitHub Actions to create and approve pull requests' (Settings > Actions > General), or open one from branch $BRANCH."
fi
