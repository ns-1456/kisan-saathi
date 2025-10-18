#!/bin/bash

# Update GitHub Milestone Deadlines for Kisan Saathi
# This script updates all milestone due dates to realistic timelines

echo "📅 Updating Sprint Deadlines for Kisan Saathi"
echo "=============================================="

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed!"
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Please authenticate with GitHub CLI first:"
    echo "Run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI authenticated"

# Calculate dates (2 weeks per sprint)
# Starting from today (October 18, 2025)
SPRINT_1_DUE="2025-11-01T23:59:59Z"  # 2 weeks from start
SPRINT_2_DUE="2025-11-15T23:59:59Z"  # 4 weeks from start
SPRINT_3_DUE="2025-11-29T23:59:59Z"  # 6 weeks from start
SPRINT_4_DUE="2025-12-13T23:59:59Z"  # 8 weeks from start
SPRINT_5_DUE="2025-12-27T23:59:59Z"  # 10 weeks from start
SPRINT_6_DUE="2026-01-10T23:59:59Z"  # 12 weeks from start
SPRINT_7_DUE="2026-01-24T23:59:59Z"  # 14 weeks from start
SPRINT_8_DUE="2026-02-07T23:59:59Z"  # 16 weeks from start

echo "📅 New Sprint Timeline:"
echo "  Sprint 1: Infrastructure Setup & Data Collection (Oct 18 - Nov 1, 2025)"
echo "  Sprint 2: Data Processing & Knowledge Base Creation (Nov 1 - Nov 15, 2025)"
echo "  Sprint 3: Base Model Selection & Baseline (Nov 15 - Nov 29, 2025)"
echo "  Sprint 4: LoRA Fine-tuning (Nov 29 - Dec 13, 2025)"
echo "  Sprint 5: RAG Pipeline Implementation (Dec 13 - Dec 27, 2025)"
echo "  Sprint 6: Model Quantization & Mobile Optimization (Dec 27, 2025 - Jan 10, 2026)"
echo "  Sprint 7: CNN for Crop Disease Detection (Jan 10 - Jan 24, 2026)"
echo "  Sprint 8: Multi-modal Integration & Final Testing (Jan 24 - Feb 7, 2026)"
echo ""

# Get milestone numbers
echo "📋 Getting milestone numbers..."
MILESTONE_1=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 1")) | .number')
MILESTONE_2=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 2")) | .number')
MILESTONE_3=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 3")) | .number')
MILESTONE_4=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 4")) | .number')
MILESTONE_5=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 5")) | .number')
MILESTONE_6=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 6")) | .number')
MILESTONE_7=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 7")) | .number')
MILESTONE_8=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 8")) | .number')

echo "📅 Current milestone numbers:"
echo "  Sprint 1: $MILESTONE_1"
echo "  Sprint 2: $MILESTONE_2"
echo "  Sprint 3: $MILESTONE_3"
echo "  Sprint 4: $MILESTONE_4"
echo "  Sprint 5: $MILESTONE_5"
echo "  Sprint 6: $MILESTONE_6"
echo "  Sprint 7: $MILESTONE_7"
echo "  Sprint 8: $MILESTONE_8"
echo ""

# Update milestone deadlines
echo "🔄 Updating milestone deadlines..."

# Update Sprint 1
echo "Updating Sprint 1 deadline to $SPRINT_1_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_1 -X PATCH -f due_on="$SPRINT_1_DUE"

# Update Sprint 2
echo "Updating Sprint 2 deadline to $SPRINT_2_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_2 -X PATCH -f due_on="$SPRINT_2_DUE"

# Update Sprint 3
echo "Updating Sprint 3 deadline to $SPRINT_3_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_3 -X PATCH -f due_on="$SPRINT_3_DUE"

# Update Sprint 4
echo "Updating Sprint 4 deadline to $SPRINT_4_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_4 -X PATCH -f due_on="$SPRINT_4_DUE"

# Update Sprint 5
echo "Updating Sprint 5 deadline to $SPRINT_5_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_5 -X PATCH -f due_on="$SPRINT_5_DUE"

# Update Sprint 6
echo "Updating Sprint 6 deadline to $SPRINT_6_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_6 -X PATCH -f due_on="$SPRINT_6_DUE"

# Update Sprint 7
echo "Updating Sprint 7 deadline to $SPRINT_7_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_7 -X PATCH -f due_on="$SPRINT_7_DUE"

# Update Sprint 8
echo "Updating Sprint 8 deadline to $SPRINT_8_DUE"
gh api repos/ns-1456/kisan-saathi/milestones/$MILESTONE_8 -X PATCH -f due_on="$SPRINT_8_DUE"

echo ""
echo "✅ All milestone deadlines updated successfully!"
echo ""
echo "📊 Updated Timeline Summary:"
echo "  🚀 Sprint 1: Infrastructure Setup & Data Collection"
echo "     Start: October 18, 2025 | End: November 1, 2025 (2 weeks)"
echo ""
echo "  📊 Sprint 2: Data Processing & Knowledge Base Creation"
echo "     Start: November 1, 2025 | End: November 15, 2025 (2 weeks)"
echo ""
echo "  🧠 Sprint 3: Base Model Selection & Baseline"
echo "     Start: November 15, 2025 | End: November 29, 2025 (2 weeks)"
echo ""
echo "  🎯 Sprint 4: LoRA Fine-tuning"
echo "     Start: November 29, 2025 | End: December 13, 2025 (2 weeks)"
echo ""
echo "  🔍 Sprint 5: RAG Pipeline Implementation"
echo "     Start: December 13, 2025 | End: December 27, 2025 (2 weeks)"
echo ""
echo "  ⚡ Sprint 6: Model Quantization & Mobile Optimization"
echo "     Start: December 27, 2025 | End: January 10, 2026 (2 weeks)"
echo ""
echo "  🌾 Sprint 7: CNN for Crop Disease Detection"
echo "     Start: January 10, 2026 | End: January 24, 2026 (2 weeks)"
echo ""
echo "  🔗 Sprint 8: Multi-modal Integration & Final Testing"
echo "     Start: January 24, 2026 | End: February 7, 2026 (2 weeks)"
echo ""
echo "🎯 Total Project Duration: 16 weeks (4 months)"
echo "📅 Project Completion: February 7, 2026"
echo ""
echo "🔗 Repository Links:"
echo "  📊 Repository: https://github.com/ns-1456/kisan-saathi"
echo "  📋 Issues: https://github.com/ns-1456/kisan-saathi/issues"
echo "  🎯 Milestones: https://github.com/ns-1456/kisan-saathi/milestones"
echo ""
echo "🚀 Ready to start Sprint 1 implementation!"
