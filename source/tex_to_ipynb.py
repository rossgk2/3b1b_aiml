#!/usr/bin/env python3
"""
LaTeX to Jupyter Notebook Converter

Converts .tex files to .ipynb files with proper handling of:
- Sections and subsections (with * preservation)
- Math environments (align, align*)
- Footnotes
- Quote and quotation environments
- Lists and other LaTeX constructs
"""

import re
import json
import argparse
import sys
from typing import List, Dict, Tuple, Any
from pathlib import Path


class LaTeXToNotebookConverter:
    def __init__(self):
        self.footnotes = {}
        self.footnote_counter = 1
        
    def convert_file(self, tex_file: str, output_file: str = None) -> str:
        """Convert a LaTeX file to Jupyter notebook format."""
        with open(tex_file, 'r', encoding='utf-8') as f:
            tex_content = f.read()
        
        # Parse the LaTeX content
        cells = self.parse_latex(tex_content)
        
        # Create the notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Determine output filename
        if output_file is None:
            output_file = Path(tex_file).with_suffix('.ipynb')
        
        # Write the notebook
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        return str(output_file)
    
    def parse_latex(self, content: str) -> List[Dict[str, Any]]:
        """Parse LaTeX content and convert to notebook cells."""
        # Remove document class and packages
        content = self.remove_preamble(content)
        
        # Extract document body
        content = self.extract_document_body(content)
        
        # Collect footnotes first
        content = self.collect_footnotes(content)
        
        # Split content into logical sections
        sections = self.split_into_sections(content)
        
        cells = []
        # Track footnotes for each section hierarchy level
        section_footnotes = {'section': [], 'subsection': [], 'subsubsection': []}
        
        for i, section in enumerate(sections):
            if section.strip():
                cell = self.create_cell(section)
                cells.append(cell)
                
                # Determine what level this section is and collect its footnotes
                section_level = self.get_section_level(section)
                footnotes_in_section = self.get_footnotes_in_section(section)
                
                # Add footnotes to appropriate level
                if footnotes_in_section:
                    section_footnotes[section_level].extend(footnotes_in_section)
                
                # Check if we need to output footnotes (at end of narrowest environment)
                next_section_level = None
                if i + 1 < len(sections):
                    next_section_level = self.get_section_level(sections[i + 1])
                
                # Output footnotes if we're at the end of a section or moving to a higher level
                footnote_cell = self.should_output_footnotes(section_level, next_section_level, section_footnotes)
                if footnote_cell:
                    cells.append(footnote_cell)
        
        # Output any remaining footnotes at the end
        remaining_footnotes = []
        for level_footnotes in section_footnotes.values():
            remaining_footnotes.extend(level_footnotes)
        if remaining_footnotes:
            cells.append(self.create_footnote_cell(remaining_footnotes))
        
        return cells
    
    def remove_preamble(self, content: str) -> str:
        """Remove LaTeX preamble (everything before \\begin{document})."""
        doc_start = content.find(r'\begin{document}')
        if doc_start != -1:
            return content[doc_start:]
        return content
    
    def split_content_into_paragraphs(self, content: str) -> List[str]:
        """Split content without section headers into paragraph-based chunks."""
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', content.strip())
        
        # Filter out empty paragraphs and group small paragraphs together
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If this paragraph is reasonably long or current section is empty, start new section
            if len(paragraph) > 200 or not current_section:
                if current_section:
                    sections.append(current_section.strip())
                current_section = paragraph
            else:
                # Add to current section
                current_section += "\n\n" + paragraph
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections if sections else [content]

    
    def section_needs_splitting(self, section: str) -> bool:
        """Check if a section is too large and needs to be split."""
        # CRITICAL: Don't split if there are list environments
        if r'\begin{itemize}' in section or r'\begin{enumerate}' in section:
            return False
        if r'\end{itemize}' in section or r'\end{enumerate}' in section:
            return False
        
        # Split if section has both a header and substantial content after it
        section_pattern = r'(\\section\*?\{[^}]+\}|\\subsection\*?\{[^}]+\}|\\subsubsection\*?\{[^}]+\})'
        match = re.search(section_pattern, section)
        
        if match:
            # Check if there's substantial content after the header
            content_after_header = section[match.end():].strip()
            # Apply consistent splitting logic: split if there's any paragraph after any header type
            paragraphs = re.split(r'\n\s*\n', content_after_header)
            return len(paragraphs) > 1 or len(content_after_header) > 100
        
        return False
    
    def split_large_section(self, section: str) -> List[str]:
        """Split a large section into header + content chunks."""
        section_pattern = r'(\\section\*?\{[^}]+\}|\\subsection\*?\{[^}]+\}|\\subsubsection\*?\{[^}]+\})'
        match = re.search(section_pattern, section)
        
        if not match:
            return [section]
        
        header = section[match.start():match.end()]
        content_after = section[match.end():].strip()
        
        # Split the content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content_after)
        
        sections = []
        
        # First section: just the header by itself
        sections.append(header)
        
        # Remaining content as separate sections, grouping small paragraphs
        current_content = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if not current_content:
                current_content = paragraph
            elif len(current_content) + len(paragraph) < 400:
                # Group small paragraphs together
                current_content += "\n\n" + paragraph
            else:
                # Current content is substantial enough, add it as a section
                sections.append(current_content)
                current_content = paragraph
        
        # Add any remaining content
        if current_content:
            sections.append(current_content)
        
        return sections

    def clean_whitespace(self, content: str) -> str:
        """Clean up tabs and excessive whitespace."""
        # Replace tabs with spaces
        content = content.replace('\t', ' ')
        
        # Remove leading whitespace from each line while preserving paragraph structure
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip leading/trailing whitespace but preserve empty lines
            cleaned_line = line.strip()
            cleaned_lines.append(cleaned_line)
        
        # Rejoin and normalize multiple empty lines
        content = '\n'.join(cleaned_lines)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content
    
    def extract_document_body(self, content: str) -> str:
        """Extract content between \\begin{document} and \\end{document}."""
        begin_match = re.search(r'\\begin\{document\}', content)
        end_match = re.search(r'\\end\{document\}', content)
        
        if begin_match and end_match:
            start = begin_match.end()
            end = end_match.start()
            body = content[start:end]
        elif begin_match:
            body = content[begin_match.end():]
        else:
            body = content
        
        # Clean up whitespace and tabs
        body = self.clean_whitespace(body)
        return body.strip()
    
    def collect_footnotes(self, content: str) -> str:
        """Collect footnotes and replace footnote marks with references."""
        # Improved pattern to handle nested braces better
        footnote_pattern = r'\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}'
        footnotetext_pattern = r'\\footnotetext\{((?:[^{}]|\{[^{}]*\})*)\}'
        footnotemark_pattern = r'\\footnotemark'
        
        def replace_footnote(match):
            footnote_text = match.group(1)
            footnote_id = f"footnote_{self.footnote_counter}"
            self.footnotes[footnote_id] = footnote_text
            footnote_number = self.footnote_counter
            self.footnote_counter += 1
            return self.number_to_superscript(footnote_number)
        
        def replace_footnotetext(match):
            footnote_text = match.group(1)
            footnote_id = f"footnote_{self.footnote_counter - 1}"  # Use the last footnotemark
            self.footnotes[footnote_id] = footnote_text
            return ""
        
        def replace_footnotemark(match):
            footnote_id = f"footnote_{self.footnote_counter}"
            footnote_number = self.footnote_counter
            self.footnote_counter += 1
            return self.number_to_superscript(footnote_number)
        
        # Process footnotes
        content = re.sub(footnote_pattern, replace_footnote, content)
        content = re.sub(footnotemark_pattern, replace_footnotemark, content)
        content = re.sub(footnotetext_pattern, replace_footnotetext, content)
        
        return content
    
    def split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections."""
        # Split on section headers and major environments
        section_pattern = r'(\\section\*?\{[^}]+\}|\\subsection\*?\{[^}]+\}|\\subsubsection\*?\{[^}]+\})'
        
        parts = re.split(section_pattern, content)
        sections = []
        
        current_section = ""
        for i, part in enumerate(parts):
            if re.match(section_pattern, part):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part
            else:
                current_section += part
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        # Post-process sections to break up overly long ones
        final_sections = []
        for section in sections:
            if self.section_needs_splitting(section):
                final_sections.extend(self.split_large_section(section))
            else:
                final_sections.append(section)
        
        # If we have no sections with headers, split the content into smaller chunks
        # to avoid having everything as one giant section
        if len(final_sections) == 1 and not re.search(section_pattern, final_sections[0]):
            return self.split_content_into_paragraphs(final_sections[0])
        
        return final_sections
    
    def create_cell(self, content: str) -> Dict[str, Any]:
        """Create a notebook cell from LaTeX content."""
        # Convert LaTeX to Markdown
        markdown_content = self.latex_to_markdown(content)
        
        # Split into lines but keep the newline character on each line except the last
        lines = markdown_content.split('\n')
        source = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        }
    
    def latex_to_markdown(self, content: str) -> str:
        """Convert LaTeX markup to Markdown."""
        # Clean whitespace first
        content = self.clean_whitespace(content)
        
        # Handle sections - but be more careful about when to apply header formatting
        # Only apply header formatting if the content actually starts with a section command
        if re.match(r'^\s*\\section\*?\{', content):
            content = re.sub(r'\\section\*?\{([^}]+)\}', r'# \1', content, count=1)
        elif re.match(r'^\s*\\subsection\*?\{', content):
            content = re.sub(r'\\subsection\*?\{([^}]+)\}', r'## \1', content, count=1)
        elif re.match(r'^\s*\\subsubsection\*?\{', content):
            content = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'### \1', content, count=1)
        
        # Handle any remaining section headers (in case there are multiple in one cell)
        content = re.sub(r'\\section\*?\{([^}]+)\}', r'# \1', content)
        content = re.sub(r'\\subsection\*?\{([^}]+)\}', r'## \1', content)
        content = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'### \1', content)
        
        # Handle text formatting
        content = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', content)
        content = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', content)
        content = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', content)
        
        # Handle quotes
        content = re.sub(r'``([^\']+)\'\'', r'"\1"', content)
        content = re.sub(r'`([^\']+)\'', r'"\1"', content)
        
        # Handle quote and quotation environments
        content = self.handle_quote_environments(content)
        
        # Handle lists
        content = self.handle_lists(content)
        
        # Handle align environments (preserve them as LaTeX)
        content = self.handle_align_environments(content)
        
        # Handle inline math (preserve as LaTeX)
        content = re.sub(r'\$([^$]+)\$', r'$\1$', content)
        
        # Handle display math (preserve as LaTeX)
        content = self.handle_display_math(content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def handle_quote_environments(self, content: str) -> str:
        """Handle quote and quotation environments."""
        # Handle \quote{} command
        content = re.sub(r'\\quote\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'> \1', content)
        
        # Handle quote environment
        quote_pattern = r'\\begin\{quote\}(.*?)\\end\{quote\}'
        def replace_quote(match):
            quote_text = match.group(1).strip()
            lines = quote_text.split('\n')
            quoted_lines = ['> ' + line if line.strip() else '>' for line in lines]
            return '\n'.join(quoted_lines)
        content = re.sub(quote_pattern, replace_quote, content, flags=re.DOTALL)
        
        # Handle quotation environment
        quotation_pattern = r'\\begin\{quotation\}(.*?)\\end\{quotation\}'
        content = re.sub(quotation_pattern, replace_quote, content, flags=re.DOTALL)
        
        return content

    def handle_lists(self, content: str) -> str:
        """Handle itemize and enumerate environments."""
        # Handle itemize
        itemize_pattern = r'\\begin\{itemize\}(.*?)\\end\{itemize\}'
        def replace_itemize(match):
            items_text = match.group(1).strip()
            # Split on \item and filter out empty items
            items = re.split(r'\\item\s+', items_text)
            markdown_items = []
            for i, item in enumerate(items):
                item = item.strip()
                if item:  # Skip empty items
                    # First split often contains content before first \item
                    if i == 0 and not items_text.strip().startswith('\\item'):
                        # This is content before the first item, skip it
                        continue
                    markdown_items.append(f"- {item}\n")
            return ''.join(markdown_items) if markdown_items else ''
        content = re.sub(itemize_pattern, replace_itemize, content, flags=re.DOTALL)
        
        # Handle enumerate
        enumerate_pattern = r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}'
        def replace_enumerate(match):
            items_text = match.group(1).strip()
            # Split on \item and filter out empty items
            items = re.split(r'\\item\s+', items_text)
            markdown_items = []
            item_num = 1
            for i, item in enumerate(items):
                item = item.strip()
                if item:  # Skip empty items
                    # First split often contains content before first \item
                    if i == 0 and not items_text.strip().startswith('\\item'):
                        # This is content before the first item, skip it
                        continue
                    markdown_items.append(f"{item_num}. {item}\n")
                    item_num += 1
            return ''.join(markdown_items) if markdown_items else ''
        content = re.sub(enumerate_pattern, replace_enumerate, content, flags=re.DOTALL)
        
        return content
    
    def handle_align_environments(self, content: str) -> str:
        """Handle align and align* environments, preserving them as LaTeX."""
        # Handle align*
        align_star_pattern = r'\\begin\{align\*\}(.*?)\\end\{align\*\}'
        def replace_align_star(match):
            align_content = match.group(1).strip()
            return f"$$\\begin{{align*}}\n{align_content}\n\\end{{align*}}$$"
        content = re.sub(align_star_pattern, replace_align_star, content, flags=re.DOTALL)
        
        # Handle align (numbered)
        align_pattern = r'\\begin\{align\}(.*?)\\end\{align\}'
        def replace_align(match):
            align_content = match.group(1).strip()
            return f"$$\\begin{{align}}\n{align_content}\n\\end{{align}}$$"
        content = re.sub(align_pattern, replace_align, content, flags=re.DOTALL)
        
        return content
    
    def handle_display_math(self, content: str) -> str:
        """Handle display math environments."""
        # Handle \[ ... \]
        content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
        
        # Handle $$ ... $$
        content = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', content, flags=re.DOTALL)
        
        return content
    
    def number_to_superscript(self, num: int) -> str:
        """Convert a number to Unicode superscript characters."""
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        }
        return ''.join(superscript_map.get(digit, digit) for digit in str(num))
    
    def toggle_italics(self, text: str) -> str:
        """Toggle italics in text - convert *text* to text and text to *text*."""
        # This will swap italicized and non-italicized text
        # Split on asterisk pairs and toggle each segment
        parts = re.split(r'(\*[^*]+\*)', text)
        result = []
        
        for part in parts:
            if re.match(r'^\*[^*]+\*$', part):
                # Remove asterisks (was italicized, now normal)
                result.append(part[1:-1])
            elif part.strip():
                # Add asterisks (was normal, now italicized)
                result.append(f'*{part}*')
            else:
                # Keep whitespace/empty parts as-is
                result.append(part)
        
        return ''.join(result)
    
    def get_section_level(self, section: str) -> str:
        """Determine the section level (section, subsection, subsubsection)."""
        if re.search(r'\\subsubsection\*?\{', section):
            return 'subsubsection'
        elif re.search(r'\\subsection\*?\{', section):
            return 'subsection'
        elif re.search(r'\\section\*?\{', section):
            return 'section'
        else:
            return 'section'  # Default to section level for content without headers
    
    def get_footnotes_in_section(self, section: str) -> List[Tuple[int, str]]:
        """Get footnotes that appear in this section."""
        footnotes_in_section = []
        
        for footnote_id, footnote_content in self.footnotes.items():
            footnote_num = int(footnote_id.split('_')[1])
            superscript_num = self.number_to_superscript(footnote_num)
            if superscript_num in section:
                footnotes_in_section.append((footnote_num, footnote_content))
        
        return footnotes_in_section
    
    def should_output_footnotes(self, current_level: str, next_level: str, section_footnotes: Dict) -> Dict[str, Any]:
        """Determine if footnotes should be output based on section hierarchy."""
        hierarchy = ['subsubsection', 'subsection', 'section']
        current_idx = hierarchy.index(current_level) if current_level in hierarchy else 2
        
        # Output footnotes if:
        # 1. We're at the end (next_level is None)
        # 2. Next section is at same level or higher (narrower or same scope)
        should_output = False
        footnotes_to_output = []
        
        if next_level is None:
            # End of document - output all remaining footnotes
            should_output = True
            for level in hierarchy:
                footnotes_to_output.extend(section_footnotes[level])
                section_footnotes[level] = []
        else:
            next_idx = hierarchy.index(next_level) if next_level in hierarchy else 2
            
            # If moving to same level or higher, output footnotes from current level and below
            if next_idx <= current_idx:
                should_output = True
                for i in range(current_idx + 1):
                    level = hierarchy[i]
                    footnotes_to_output.extend(section_footnotes[level])
                    section_footnotes[level] = []
        
        if should_output and footnotes_to_output:
            return self.create_footnote_cell(footnotes_to_output)
        return None
    
    def create_footnote_cell(self, footnotes: List[Tuple[int, str]]) -> Dict[str, Any]:
        """Create a footnote cell with proper formatting."""
        if not footnotes:
            return None
        
        lines = ["---\n"]  # Horizontal rule with newline for proper rendering
        
        for footnote_num, footnote_content in footnotes:
            # Convert LaTeX formatting in footnotes
            footnote_content = self.latex_to_markdown(footnote_content)
            # Remove italic markers
            footnote_content = footnote_content.replace('*', '')
            # Format: superscript number + entire content wrapped in italics
            superscript_num = self.number_to_superscript(footnote_num)
            lines.append(f"{superscript_num} *{footnote_content}*")
        
        return {
            "cell_type": "markdown", 
            "metadata": {},
            "source": lines
        }

def main():
    parser = argparse.ArgumentParser(description='Convert LaTeX files to Jupyter notebooks')
    parser.add_argument('input_file', help='Input LaTeX file (.tex)')
    parser.add_argument('-o', '--output', help='Output notebook file (.ipynb)')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    converter = LaTeXToNotebookConverter()
    
    try:
        output_file = converter.convert_file(args.input_file, args.output)
        print(f"Successfully converted '{args.input_file}' to '{output_file}'")
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()