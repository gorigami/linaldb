#!/bin/bash

# Configuration script for LINALDB GitHub Repository Rulesets
# Requires: gh CLI authenticated with 'repo' scope

REPO="gorigamidev/linaldb"

echo "🚀 Configuring Repository Rulesets for $REPO..."

# Define the ruleset JSON
RULESET_JSON=$(cat <<EOF
{
  "name": "Main Protection",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [
    {
      "actor_id": 5,
      "actor_type": "RepositoryRole",
      "bypass_mode": "always"
    }
  ],
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_signatures" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          { "context": "Test" },
          { "context": "Format" },
          { "context": "Clippy" },
          { "context": "Smoke Test" }
        ]
      }
    }
  ]
}
EOF
)

# Find existing ruleset ID
RULESET_ID=$(gh api /repos/$REPO/rulesets --template '{{range .}}{{if eq .name "Main Protection"}}{{printf "%.0f" .id}}{{end}}{{end}}')

if [ -z "$RULESET_ID" ]; then
    echo "Creating ruleset via GitHub API..."
    echo "$RULESET_JSON" | gh api --method POST /repos/$REPO/rulesets --input -
else
    echo "Updating existing ruleset $RULESET_ID via GitHub API..."
    echo "$RULESET_JSON" | gh api --method PUT /repos/$REPO/rulesets/$RULESET_ID --input -
fi

echo "✅ Configuration complete."
echo "Visit https://github.com/$REPO/settings/rules to verify."
