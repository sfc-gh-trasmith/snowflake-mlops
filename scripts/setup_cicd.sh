#!/bin/bash
# Configure GitHub Actions CI/CD access to Snowflake
#
# This script:
#   1. Creates a network policy allowing GitHub-hosted runner IPs
#      (using Snowflake's managed network rule -- auto-tracks IP ranges)
#   2. Creates a service user with OIDC workload identity
#   3. Assigns the network policy to the service user
#
# Prerequisites:
#   - ACCOUNTADMIN role
#   - Update SUBJECT below with your repo's owner/repo IDs
#
# Usage: bash scripts/setup_cicd.sh

set -e

# --- CONFIGURATION ---
# Update these with your GitHub repo details.
# GitHub enriches OIDC subjects with numeric IDs: owner@<owner_id>/repo@<repo_id>
# Find yours in the Actions error log on first run, or via GitHub API.
GITHUB_OWNER="sfc-gh-trasmith"
GITHUB_OWNER_ID="256389544"
GITHUB_REPO="snowflake-mlops"
GITHUB_REPO_ID="1306141664"

SUBJECT="repo:${GITHUB_OWNER}@${GITHUB_OWNER_ID}/${GITHUB_REPO}@${GITHUB_REPO_ID}:ref:refs/heads/main"
SERVICE_USER="SVC_GITHUB_ACTIONS"

echo "=== Setting up GitHub Actions CI/CD access ==="
echo "  Service user: ${SERVICE_USER}"
echo "  OIDC subject: ${SUBJECT}"
echo ""

snow sql -q "
-- =============================================================================
-- Step 1: Network Policy for GitHub Actions
-- Uses Snowflake's managed network rule that auto-tracks GitHub runner IPs.
-- This avoids hardcoding CIDR ranges that change frequently.
-- =============================================================================
CREATE NETWORK RULE IF NOT EXISTS GITHUB_ACTIONS_NETWORK_RULE
    MODE = INGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('github.com');

CREATE NETWORK POLICY IF NOT EXISTS GITHUB_ACTIONS_POLICY
    ALLOWED_NETWORK_RULE_LIST = ('SNOWFLAKE.NETWORK_SECURITY.GITHUBACTIONS_GLOBAL')
    COMMENT = 'Allow GitHub Actions runners via managed network rule';

-- =============================================================================
-- Step 2: Service User with OIDC Workload Identity
-- GitHub mints short-lived OIDC tokens; Snowflake validates issuer + subject.
-- No secrets stored in GitHub -- zero credential management.
-- =============================================================================
CREATE USER IF NOT EXISTS ${SERVICE_USER}
    TYPE = SERVICE
    DEFAULT_ROLE = PUBLIC
    COMMENT = 'GitHub Actions CI/CD service user (OIDC)'
    WORKLOAD_IDENTITY = (
        TYPE = OIDC
        ISSUER = 'https://token.actions.githubusercontent.com'
        SUBJECT = '${SUBJECT}'
    );

-- Grant deployment role
GRANT ROLE ACCOUNTADMIN TO USER ${SERVICE_USER};

-- =============================================================================
-- Step 3: Assign Network Policy to Service User
-- This overrides the account-level network policy for this user only,
-- allowing GitHub Actions runners through while keeping other restrictions.
-- =============================================================================
ALTER USER ${SERVICE_USER} SET NETWORK_POLICY = 'GITHUB_ACTIONS_POLICY';
"

echo ""
echo "=== CI/CD access configured ==="
echo ""
echo "Next steps:"
echo "  1. Push a workflow using snowflakedb/snowflake-actions@v3 with use-oidc: true"
echo "  2. Set SNOWFLAKE_ACCOUNT as a GitHub variable or in workflow env"
echo "  3. On first run, if the subject doesn't match, check the error log"
echo "     for the actual subject claim and update the user's WORKLOAD_IDENTITY"
echo ""
echo "To find your GitHub owner/repo IDs:"
echo "  curl -s https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO} | jq '.id, .owner.id'"

# =============================================================================
# Step 4: GitHub Branch Protection & Repo Settings
# Requires: gh CLI authenticated with repo admin access
# =============================================================================
echo ""
echo "=== Setting up GitHub branch protection ==="

# Require PRs to main (no direct pushes), status checks must pass, 1 reviewer
gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/branches/main/protection" -X PUT --input - <<'PROTECTION'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint-and-test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
PROTECTION

# Create GitHub Environments with protection rules
gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/environments/STAGE" -X PUT --input - <<< '{}'
gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/environments/PROD" -X PUT --input - <<< '{"reviewers":[{"type":"User","id":'"$(gh api user --jq '.id')"'}]}'

echo ""
echo "=== Branch protection configured ==="
echo "  - PRs required to merge to main (no direct pushes)"
echo "  - Status check 'lint-and-test' must pass"
echo "  - 1 approving review required"
echo "  - Stale reviews dismissed on new pushes"
echo "  - Force pushes and branch deletion blocked"
echo "  - PROD environment requires reviewer approval"
