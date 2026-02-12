#!/bin/bash

# Simple script to create PDFs from markdown using pandoc
# If pandoc is not installed, this will show instructions

echo "Creating elaborate PDF guide..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "❌ pandoc not found. Installing with homebrew..."
    brew install pandoc
fi

# Convert Part 1
echo "Converting Part 1..."
pandoc DETAILED_GUIDE_FOR_UNDERSTANDING.md \
    -o "Recommendation_System_Detailed_Guide_Part1.pdf" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc \
    2>/dev/null || echo "⚠️  Part 1 PDF creation requires LaTeX (install with: brew install mactex)"

# Convert Part 2
echo "Converting Part 2..."
pandoc DETAILED_GUIDE_PART2.md \
    -o "Recommendation_System_Detailed_Guide_Part2.pdf" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc \
    2>/dev/null || echo "⚠️  Part 2 PDF creation requires LaTeX (install with: brew install mactex)"

# Also keep the simple technical PDF
echo "Keeping original technical PDF..."

echo ""
echo "✅ Documentation created:"
echo "   📄 COMPLETE_SYSTEM_FLOW.md (technical overview)"
echo "   📄 DETAILED_GUIDE_FOR_UNDERSTANDING.md (beginner-friendly Part 1)"
echo "   📄 DETAILED_GUIDE_PART2.md (beginner-friendly Part 2)"
echo "   📄 Recommendation_System_Complete_Flow.pdf (technical PDF)"
echo ""
echo "📖 To read the guides:"
echo "   open DETAILED_GUIDE_FOR_UNDERSTANDING.md"
echo "   open DETAILED_GUIDE_PART2.md"
echo ""
echo "Or view the PDF:"
echo "   open Recommendation_System_Complete_Flow.pdf"
