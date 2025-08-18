#!/usr/bin/env python3
"""
Standalone Document Chunk Size Analyzer

This program analyzes any document and provides intelligent recommendations
for optimal chunk size and overlap settings for vector embeddings.

Usage:
    python chunk_analyzer.py

Requirements:
    pip install pandas numpy matplotlib seaborn PyPDF2
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import statistics


# File reading utilities
def read_file(file_path: str) -> str:
    """Read content from various file types."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == '.pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF files. Install with: pip install PyPDF2")

    elif file_path.suffix.lower() in ['.txt', '.md', '.rst']:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    elif file_path.suffix.lower() == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()

    else:
        # Try to read as text
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()


class DocumentAnalyzer:
    """Comprehensive document analysis for chunking optimization."""

    def __init__(self, text: str):
        self.text = text
        self.analysis = {}
        self._analyze_document()

    def _analyze_document(self):
        """Perform comprehensive document analysis."""

        # Basic metrics
        self.analysis['total_characters'] = len(self.text)
        self.analysis['total_words'] = len(self.text.split())

        # Sentence analysis
        sentences = self._extract_sentences()
        self.analysis['total_sentences'] = len(sentences)
        self.analysis['sentence_lengths'] = [len(s) for s in sentences if s.strip()]

        # Paragraph analysis
        paragraphs = self._extract_paragraphs()
        self.analysis['total_paragraphs'] = len(paragraphs)
        self.analysis['paragraph_lengths'] = [len(p) for p in paragraphs if p.strip()]

        # Line analysis
        lines = self.text.split('\n')
        self.analysis['total_lines'] = len(lines)
        self.analysis['line_lengths'] = [len(line) for line in lines if line.strip()]

        # Content type detection
        self.analysis['content_type'] = self._detect_content_type()

        # Structure analysis
        self.analysis['structure_score'] = self._analyze_structure()

        # Density analysis
        self.analysis['information_density'] = self._calculate_density()

    def _extract_sentences(self) -> List[str]:
        """Extract sentences using multiple delimiters."""
        # Simple sentence splitting - can be improved with NLTK
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _extract_paragraphs(self) -> List[str]:
        """Extract paragraphs."""
        paragraphs = re.split(r'\n\s*\n', self.text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 20]

    def _detect_content_type(self) -> str:
        """Detect the type of content."""
        text_lower = self.text.lower()

        # Count different indicators
        code_indicators = len(re.findall(r'(def |class |import |function|var |const |#include)', text_lower))
        academic_indicators = len(
            re.findall(r'(abstract|introduction|methodology|conclusion|references|bibliography)', text_lower))
        narrative_indicators = len(re.findall(r'(chapter|once upon|story|character|dialogue)', text_lower))
        technical_indicators = len(
            re.findall(r'(algorithm|procedure|step|process|method|system|api|database)', text_lower))
        legal_indicators = len(re.findall(r'(whereas|therefore|pursuant|agreement|contract|terms)', text_lower))

        # Determine type based on indicators
        if code_indicators > 5:
            return "code_documentation"
        elif academic_indicators > 3:
            return "academic_paper"
        elif legal_indicators > 3:
            return "legal_document"
        elif narrative_indicators > 2:
            return "narrative"
        elif technical_indicators > 5:
            return "technical_documentation"
        else:
            return "general"

    def _analyze_structure(self) -> float:
        """Analyze document structure (0-1 score, higher = more structured)."""
        structure_score = 0.0

        # Headers/titles (markdown style or caps)
        headers = len(re.findall(r'^#{1,6}\s|\n[A-Z][A-Z\s]{5,}\n', self.text, re.MULTILINE))
        structure_score += min(headers / 10, 0.3)

        # Lists
        lists = len(re.findall(r'^\s*[-*+•]\s|^\s*\d+\.\s', self.text, re.MULTILINE))
        structure_score += min(lists / 20, 0.2)

        # Consistent paragraph lengths
        if self.analysis['paragraph_lengths']:
            para_std = np.std(self.analysis['paragraph_lengths'])
            para_mean = np.mean(self.analysis['paragraph_lengths'])
            if para_mean > 0:
                consistency = 1 - min(para_std / para_mean, 1)
                structure_score += consistency * 0.3

        # Sentence length consistency
        if self.analysis['sentence_lengths']:
            sent_std = np.std(self.analysis['sentence_lengths'])
            sent_mean = np.mean(self.analysis['sentence_lengths'])
            if sent_mean > 0:
                consistency = 1 - min(sent_std / sent_mean, 1)
                structure_score += consistency * 0.2

        return min(structure_score, 1.0)

    def _calculate_density(self) -> float:
        """Calculate information density (words per character)."""
        if self.analysis['total_characters'] > 0:
            return self.analysis['total_words'] / self.analysis['total_characters']
        return 0.0


class ChunkingRecommendationEngine:
    """Engine to generate chunking recommendations based on document analysis."""

    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.recommendations = {}
        self._generate_recommendations()

    def _generate_recommendations(self):
        """Generate comprehensive chunking recommendations."""
        analysis = self.analyzer.analysis

        # Base recommendations by content type
        base_configs = self._get_base_configurations()

        # Adjust based on document characteristics
        adjusted_configs = self._adjust_for_characteristics(base_configs)

        # Generate multiple scenarios
        scenarios = self._generate_scenarios(adjusted_configs)

        # Score and rank scenarios
        scored_scenarios = self._score_scenarios(scenarios)

        # Select best recommendation
        self.recommendations = self._select_best_recommendation(scored_scenarios)

    def _get_base_configurations(self) -> Dict[str, Tuple[int, float]]:
        """Get base configurations for different content types."""
        content_type = self.analyzer.analysis['content_type']

        configs = {
            "code_documentation": (600, 0.25),
            "academic_paper": (1000, 0.20),
            "legal_document": (800, 0.30),
            "narrative": (1200, 0.15),
            "technical_documentation": (900, 0.22),
            "general": (800, 0.20)
        }

        return {
            "conservative": configs.get(content_type, (800, 0.20)),
            "balanced": (configs.get(content_type, (800, 0.20))[0], configs.get(content_type, (800, 0.20))[1]),
            "aggressive": (int(configs.get(content_type, (800, 0.20))[0] * 0.75),
                           configs.get(content_type, (800, 0.20))[1] + 0.05)
        }

    def _adjust_for_characteristics(self, base_configs: Dict) -> Dict:
        """Adjust configurations based on document characteristics."""
        analysis = self.analyzer.analysis

        # Calculate adjustment factors
        adjustments = {}

        # Sentence length adjustment
        if analysis['sentence_lengths']:
            avg_sentence = np.mean(analysis['sentence_lengths'])
            if avg_sentence < 50:  # Short sentences
                adjustments['sentence_factor'] = 0.8
            elif avg_sentence > 150:  # Long sentences
                adjustments['sentence_factor'] = 1.3
            else:
                adjustments['sentence_factor'] = 1.0
        else:
            adjustments['sentence_factor'] = 1.0

        # Paragraph length adjustment
        if analysis['paragraph_lengths']:
            avg_paragraph = np.mean(analysis['paragraph_lengths'])
            if avg_paragraph < 200:  # Short paragraphs
                adjustments['paragraph_factor'] = 0.9
            elif avg_paragraph > 1000:  # Long paragraphs
                adjustments['paragraph_factor'] = 1.2
            else:
                adjustments['paragraph_factor'] = 1.0
        else:
            adjustments['paragraph_factor'] = 1.0

        # Structure adjustment
        structure_score = analysis['structure_score']
        if structure_score > 0.7:  # Highly structured
            adjustments['structure_factor'] = 1.1
            adjustments['overlap_adjustment'] = -0.05
        elif structure_score < 0.3:  # Poorly structured
            adjustments['structure_factor'] = 0.9
            adjustments['overlap_adjustment'] = +0.05
        else:
            adjustments['structure_factor'] = 1.0
            adjustments['overlap_adjustment'] = 0.0

        # Apply adjustments
        adjusted_configs = {}
        for scenario, (chunk_size, overlap) in base_configs.items():
            total_size_factor = (adjustments['sentence_factor'] *
                                 adjustments['paragraph_factor'] *
                                 adjustments['structure_factor'])

            new_chunk_size = int(chunk_size * total_size_factor)
            new_overlap = max(0.05, min(0.4, overlap + adjustments['overlap_adjustment']))

            adjusted_configs[scenario] = (new_chunk_size, new_overlap)

        return adjusted_configs

    def _generate_scenarios(self, base_configs: Dict) -> List[Dict]:
        """Generate multiple chunking scenarios."""
        scenarios = []

        for scenario_name, (chunk_size, overlap_ratio) in base_configs.items():
            # Generate variations
            size_variations = [
                int(chunk_size * 0.8),
                chunk_size,
                int(chunk_size * 1.2)
            ]

            overlap_variations = [
                max(0.05, overlap_ratio - 0.05),
                overlap_ratio,
                min(0.35, overlap_ratio + 0.05)
            ]

            for size in size_variations:
                for overlap in overlap_variations:
                    scenarios.append({
                        'name': f"{scenario_name}_{size}_{int(overlap * 100)}",
                        'chunk_size': size,
                        'overlap_ratio': overlap,
                        'overlap_chars': int(size * overlap),
                        'base_scenario': scenario_name
                    })

        return scenarios

    def _score_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """Score each scenario based on document characteristics."""
        analysis = self.analyzer.analysis

        for scenario in scenarios:
            score = 0.0

            # Size appropriateness score
            chunk_size = scenario['chunk_size']
            if analysis['paragraph_lengths']:
                avg_para = np.mean(analysis['paragraph_lengths'])
                if 0.5 * avg_para <= chunk_size <= 2.0 * avg_para:
                    score += 25
                elif 0.3 * avg_para <= chunk_size <= 3.0 * avg_para:
                    score += 15
                else:
                    score += 5

            # Overlap appropriateness score
            overlap_ratio = scenario['overlap_ratio']
            content_type = analysis['content_type']

            if content_type in ['academic_paper', 'technical_documentation']:
                if 0.15 <= overlap_ratio <= 0.25:
                    score += 20
            elif content_type == 'narrative':
                if 0.10 <= overlap_ratio <= 0.20:
                    score += 20
            elif content_type == 'code_documentation':
                if 0.20 <= overlap_ratio <= 0.30:
                    score += 20

            # Efficiency score (fewer chunks is generally better)
            estimated_chunks = analysis['total_characters'] / chunk_size
            if 10 <= estimated_chunks <= 100:
                score += 15
            elif 5 <= estimated_chunks <= 200:
                score += 10

            # Structure compatibility score
            structure_score = analysis['structure_score']
            if structure_score > 0.7 and chunk_size >= 800:
                score += 10
            elif structure_score < 0.3 and chunk_size <= 600:
                score += 10

            scenario['score'] = score

        return sorted(scenarios, key=lambda x: x['score'], reverse=True)

    def _select_best_recommendation(self, scored_scenarios: List[Dict]) -> Dict:
        """Select the best recommendation and generate explanation."""
        best = scored_scenarios[0]
        alternatives = scored_scenarios[1:6]  # Top 5 alternatives

        return {
            'recommended': best,
            'alternatives': alternatives,
            'all_scenarios': scored_scenarios
        }


class ChunkingNarrator:
    """Generate comprehensive narrative explanations for chunking recommendations."""

    def __init__(self, analyzer: DocumentAnalyzer, recommendations: Dict):
        self.analyzer = analyzer
        self.recommendations = recommendations

    def generate_full_report(self) -> str:
        """Generate a comprehensive chunking analysis report."""
        report = []

        # Header
        report.append("🔍 DOCUMENT CHUNKING ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        # Document overview
        report.extend(self._generate_document_overview())
        report.append("")

        # Analysis findings
        report.extend(self._generate_analysis_findings())
        report.append("")

        # Recommendation
        report.extend(self._generate_recommendation())
        report.append("")

        # Alternative options
        report.extend(self._generate_alternatives())
        report.append("")

        # Implementation guidance
        report.extend(self._generate_implementation_guidance())

        return "\n".join(report)

    def _generate_document_overview(self) -> List[str]:
        """Generate document overview section."""
        analysis = self.analyzer.analysis

        overview = ["📋 DOCUMENT OVERVIEW"]
        overview.append("-" * 20)
        overview.append(f"📄 Content Type: {analysis['content_type'].replace('_', ' ').title()}")
        overview.append(f"📊 Total Characters: {analysis['total_characters']:,}")
        overview.append(f"📝 Total Words: {analysis['total_words']:,}")
        overview.append(f"📑 Paragraphs: {analysis['total_paragraphs']}")
        overview.append(
            f"🏗️ Structure Score: {analysis['structure_score']:.2f}/1.00 {'(Well-structured)' if analysis['structure_score'] > 0.7 else '(Moderately structured)' if analysis['structure_score'] > 0.4 else '(Loosely structured)'}")

        if analysis['paragraph_lengths']:
            avg_para = np.mean(analysis['paragraph_lengths'])
            overview.append(f"📏 Average Paragraph Length: {avg_para:.0f} characters")

        if analysis['sentence_lengths']:
            avg_sent = np.mean(analysis['sentence_lengths'])
            overview.append(f"📐 Average Sentence Length: {avg_sent:.0f} characters")

        return overview

    def _generate_analysis_findings(self) -> List[str]:
        """Generate analysis findings section."""
        analysis = self.analyzer.analysis
        findings = ["🔎 KEY FINDINGS"]
        findings.append("-" * 15)

        # Content type insights
        content_type = analysis['content_type']
        if content_type == 'academic_paper':
            findings.append(
                "📚 Academic content detected - requires balanced chunks to preserve context while maintaining searchability")
        elif content_type == 'technical_documentation':
            findings.append("⚙️ Technical documentation detected - needs precise chunking to keep procedures intact")
        elif content_type == 'narrative':
            findings.append(
                "📖 Narrative content detected - larger chunks preserve story flow and character development")
        elif content_type == 'code_documentation':
            findings.append(
                "💻 Code documentation detected - smaller chunks with higher overlap for better code example retrieval")
        elif content_type == 'legal_document':
            findings.append("⚖️ Legal document detected - high overlap essential to maintain clause relationships")
        else:
            findings.append("📄 General content detected - using balanced approach for optimal performance")

        # Structure insights
        structure_score = analysis['structure_score']
        if structure_score > 0.7:
            findings.append("🏗️ Highly structured document - can use larger chunks due to clear organization")
        elif structure_score > 0.4:
            findings.append("🔧 Moderately structured document - standard chunking approach recommended")
        else:
            findings.append("🌊 Loosely structured document - smaller chunks with higher overlap for better retrieval")

        # Size insights
        if analysis['paragraph_lengths']:
            avg_para = np.mean(analysis['paragraph_lengths'])
            if avg_para < 200:
                findings.append("📏 Short paragraphs detected - chunk size adjusted downward to respect natural breaks")
            elif avg_para > 800:
                findings.append("📏 Long paragraphs detected - chunk size increased to avoid mid-paragraph splits")

        return findings

    def _generate_recommendation(self) -> List[str]:
        """Generate main recommendation section."""
        best = self.recommendations['recommended']

        rec = ["🎯 RECOMMENDED CONFIGURATION"]
        rec.append("-" * 30)
        rec.append(f"📦 Chunk Size: {best['chunk_size']} characters")
        rec.append(f"🔄 Overlap: {best['overlap_chars']} characters ({best['overlap_ratio']:.1%})")
        rec.append(f"⭐ Confidence Score: {best['score']:.0f}/100")
        rec.append("")

        # Explanation
        rec.append("💡 WHY THIS CONFIGURATION:")

        # Chunk size reasoning
        if best['chunk_size'] <= 600:
            rec.append(f"   • Small chunks ({best['chunk_size']} chars) chosen for precise information retrieval")
        elif best['chunk_size'] <= 1000:
            rec.append(f"   • Medium chunks ({best['chunk_size']} chars) chosen to balance context and precision")
        else:
            rec.append(f"   • Large chunks ({best['chunk_size']} chars) chosen to preserve context and narrative flow")

        # Overlap reasoning
        if best['overlap_ratio'] <= 0.15:
            rec.append(f"   • Low overlap ({best['overlap_ratio']:.1%}) for efficient storage with minimal redundancy")
        elif best['overlap_ratio'] <= 0.25:
            rec.append(
                f"   • Moderate overlap ({best['overlap_ratio']:.1%}) to ensure important information isn't lost at boundaries")
        else:
            rec.append(
                f"   • High overlap ({best['overlap_ratio']:.1%}) to maintain context continuity and relationship preservation")

        # Expected results
        total_chars = self.analyzer.analysis['total_characters']
        estimated_chunks = total_chars / best['chunk_size']
        rec.append("")
        rec.append("📈 EXPECTED RESULTS:")
        rec.append(f"   • Estimated chunks: ~{estimated_chunks:.0f}")
        rec.append(f"   • Storage efficiency: {1 / best['overlap_ratio']:.1f}x redundancy factor")
        rec.append(f"   • Best for: {self._get_use_case_recommendation(best)}")

        return rec

    def _generate_alternatives(self) -> List[str]:
        """Generate alternatives section."""
        alternatives = self.recommendations['alternatives']

        alt = ["🔄 ALTERNATIVE CONFIGURATIONS"]
        alt.append("-" * 30)

        for i, config in enumerate(alternatives[:3], 1):
            alt.append(
                f"{i}. Chunk Size: {config['chunk_size']}, Overlap: {config['overlap_chars']} ({config['overlap_ratio']:.1%}) - Score: {config['score']:.0f}")
            alt.append(f"   Use when: {self._get_alternative_use_case(config, i)}")
            alt.append("")

        return alt

    def _generate_implementation_guidance(self) -> List[str]:
        """Generate implementation guidance section."""
        best = self.recommendations['recommended']

        guide = ["🚀 IMPLEMENTATION GUIDANCE"]
        guide.append("-" * 25)
        guide.append("Sample LangChain configuration:")
        guide.append("")
        guide.append("```python")
        guide.append("from langchain.text_splitter import RecursiveCharacterTextSplitter")
        guide.append("")
        guide.append("text_splitter = RecursiveCharacterTextSplitter(")
        guide.append(f"    chunk_size={best['chunk_size']},")
        guide.append(f"    chunk_overlap={best['overlap_chars']},")
        guide.append("    length_function=len,")
        guide.append(")")
        guide.append("```")
        guide.append("")

        guide.append("📋 TESTING RECOMMENDATIONS:")
        guide.append("1. Start with the recommended configuration")
        guide.append("2. Test with your specific queries")
        guide.append("3. Monitor retrieval quality metrics")
        guide.append("4. Adjust based on actual performance")
        guide.append("")

        guide.append("⚠️ CONSIDERATIONS:")
        guide.append(f"• Embedding model context window: Ensure {best['chunk_size']} chars fit your model")
        guide.append("• Storage costs: Higher overlap = more storage required")
        guide.append("• Query types: Adjust chunk size based on expected question complexity")

        return guide

    def _get_use_case_recommendation(self, config: Dict) -> str:
        """Get use case recommendation for a configuration."""
        chunk_size = config['chunk_size']
        overlap_ratio = config['overlap_ratio']

        if chunk_size <= 500:
            return "precise fact retrieval, FAQ systems"
        elif chunk_size <= 800:
            return "general Q&A, search applications"
        elif chunk_size <= 1200:
            return "detailed explanations, summarization"
        else:
            return "narrative understanding, context-heavy applications"

    def _get_alternative_use_case(self, config: Dict, index: int) -> str:
        """Get use case for alternative configurations."""
        cases = [
            "you need higher precision with more focused chunks",
            "you prioritize storage efficiency over overlap",
            "you need maximum context preservation"
        ]
        return cases[min(index - 1, len(cases) - 1)]


def main():
    """Main program execution."""
    print("🔍 Document Chunking Analyzer")
    print("=" * 40)
    print()

    # Get file path from user
    file_path = input("📁 Enter the path to your document: ").strip().strip('"\'')

    if not file_path:
        print("❌ No file path provided. Exiting.")
        return

    try:
        # Read the document
        print("📖 Reading document...")
        text = read_file(file_path)

        if len(text) < 100:
            print("⚠️ Document seems very short. Analysis may not be meaningful.")

        # Analyze the document
        print("🔍 Analyzing document structure...")
        analyzer = DocumentAnalyzer(text)

        # Generate recommendations
        print("🎯 Generating chunking recommendations...")
        recommendation_engine = ChunkingRecommendationEngine(analyzer)

        # Create narrative report
        print("📝 Creating detailed report...")
        narrator = ChunkingNarrator(analyzer, recommendation_engine.recommendations)

        # Generate and display report
        report = narrator.generate_full_report()
        print("\n" + report)

        # Option to save report
        save_option = input("\n💾 Save report to file? (y/n): ").strip().lower()
        if save_option == 'y':
            output_file = f"chunking_analysis_{Path(file_path).stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to: {output_file}")

        # Show visualization option
        viz_option = input("📊 Generate visualization? (y/n): ").strip().lower()
        if viz_option == 'y':
            create_visualization(recommendation_engine.recommendations)

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        print("Please check your file and try again.")


def create_visualization(recommendations: Dict):
    """Create visualization of chunking options."""
    try:
        scenarios = recommendations['all_scenarios']
        df = pd.DataFrame(scenarios)

        plt.figure(figsize=(12, 8))

        # Create scatter plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(df['chunk_size'], df['overlap_ratio'],
                              c=df['score'], s=60, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Score')
        plt.xlabel('Chunk Size (characters)')
        plt.ylabel('Overlap Ratio')
        plt.title('Chunking Configuration Scores')

        # Highlight best option
        best = recommendations['recommended']
        plt.scatter(best['chunk_size'], best['overlap_ratio'],
                    c='red', s=100, marker='*', label='Recommended')
        plt.legend()

        # Score distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['score'], bins=15, alpha=0.7, color='skyblue')
        plt.axvline(best['score'], color='red', linestyle='--', label='Recommended Score')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()

        # Chunk size vs estimated chunks
        plt.subplot(2, 2, 3)
        estimated_chunks = [10000 / size for size in df['chunk_size']]  # Rough estimate
        plt.scatter(df['chunk_size'], estimated_chunks, alpha=0.6)
        plt.xlabel('Chunk Size')
        plt.ylabel('Estimated Number of Chunks')
        plt.title('Chunk Size vs Number of Chunks')

        # Overlap impact
        plt.subplot(2, 2, 4)
        plt.scatter(df['overlap_ratio'], df['score'], alpha=0.6)
        plt.xlabel('Overlap Ratio')
        plt.ylabel('Score')
        plt.title('Overlap Impact on Score')

        plt.tight_layout()
        plt.savefig('chunking_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Visualization saved as 'chunking_analysis.png'")

    except ImportError:
        print("📊 Visualization requires matplotlib and seaborn. Install with:")
        print("pip install matplotlib seaborn")
    except Exception as e:
        print(f"📊 Visualization error: {e}")


if __name__ == "__main__":
    main()