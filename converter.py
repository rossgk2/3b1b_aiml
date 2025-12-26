"""
Run `python converter.py --help for usage.

This converter needs the following improvements:
- Always collect all footnotes from the same LATEX section together, at the end of the corresponding Jupyter section (currently, the converter might not group all footnotes from the same LATEX section in the same Jupyter section)
- Put a horizontal bar ("---") above such footnote groupings
- In such a footnote grouping, separate individual footnotes with "  \n" (newline character preceded by two spaces)

"""

import re
import json
import argparse
from pathlib import Path

def convert_tex_to_ipynb(tex_content):
    """
    Convert LaTeX content to Jupyter notebook format.
    """
    # Initialize notebook structure
    notebook = {
        "cells": [],
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
    
    # Remove LaTeX preamble and document wrapper
    content = tex_content
    
    # Remove everything before \begin{document}
    if '\\begin{document}' in content:
        content = content.split('\\begin{document}', 1)[1]
    
    # Remove \end{document}
    if '\\end{document}' in content:
        content = content.split('\\end{document}', 1)[0]
    
    # Split content into sections
    sections = split_into_sections(content)
    
    for section in sections:
        if section.strip():
            cell = create_markdown_cell(section)
            if cell:
                notebook["cells"].append(cell)
    
    return notebook

def split_into_sections(content):
    """
    Split content into logical sections based on LaTeX structure.
    """
    sections = []
    current_section = ""
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Check if this line starts a new section
        if (line.startswith('\\section*{') or 
            line.startswith('\\subsection*{') or 
            line.startswith('\\subsubsection*{')):
            
            # Save previous section if it exists
            if current_section.strip():
                sections.append(current_section.strip())
            
            # Start new section
            current_section = line + '\n'
        else:
            current_section += line + '\n'
    
    # Add the last section
    if current_section.strip():
        sections.append(current_section.strip())
    
    return sections

def create_markdown_cell(content):
    """
    Convert LaTeX section to markdown cell content.
    """
    if not content.strip():
        return None
    
    # Convert LaTeX to markdown
    markdown_content = convert_latex_to_markdown(content)
    
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": markdown_content.split('\n')
    }
    
    return cell

def convert_latex_to_markdown(content):
    """
    Convert LaTeX syntax to markdown syntax.
    """
    # Convert sections (both numbered and unnumbered)
    content = convert_sections(content)
    
    # Convert emphasis
    content = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', content)
    content = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', content)
    
    # Convert footnotes
    content = convert_footnotes(content)
    
    # Convert quotes
    content = re.sub(r'``([^\'\']+)\'\'', r'"\1"', content)
    content = re.sub(r'`([^\']+)\'', r'"\1"', content)
    
    # Convert math environments
    content = convert_math_environments(content)
    
    # Convert lists
    content = convert_itemize(content)
    content = convert_enumerate(content)
    
    # Convert quote environment
    content = convert_quotes(content)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    return content.strip()

def convert_math_environments(content):
    """
    Convert LaTeX math environments to markdown math.
    """
    # Convert align* environments - keep them as align* for proper alignment
    def convert_align_star(match):
        math_content = match.group(1).strip()
        return f'$$\\begin{{align*}}\n{math_content}\n\\end{{align*}}$$'
    
    content = re.sub(r'\\begin\{align\*\}(.*?)\\end\{align\*\}', 
                    convert_align_star, content, flags=re.DOTALL)
    
    # Convert regular align environments to align* (since we don't want equation numbering in notebooks)
    def convert_align(match):
        math_content = match.group(1).strip()
        return f'$$\\begin{{align*}}\n{math_content}\n\\end{{align*}}$$'
    
    content = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', 
                    convert_align, content, flags=re.DOTALL)
    
    # Handle cases environment and other math constructs
    content = re.sub(r'\\text\{where \}([^$]+)', r'$\\text{where } \1$', content)
    
    # Convert simple math expressions (inline and display)
    content = re.sub(r'\$([^$]+)\$', r'$\1$', content)
    content = re.sub(r'\$\$([^$]+)\$\$', r'$$\1$$', content)
    
    return content

def convert_itemize(content):
    """
    Convert LaTeX itemize environment to markdown bulleted lists.
    """
    def convert_itemize_block(match):
        items_content = match.group(1)
        items = re.findall(r'\\item\s+([^\\]*?)(?=\\item|\Z)', items_content, re.DOTALL)
        markdown_items = []
        for item in items:
            item = item.strip()
            if item:
                # Clean up the item content
                item = re.sub(r'\s+', ' ', item)
                markdown_items.append(f'- {item}')
        return '\n'.join(markdown_items)
    
    content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', 
                    convert_itemize_block, content, flags=re.DOTALL)
    
    return content

def convert_enumerate(content):
    """
    Convert LaTeX enumerate environment to markdown numbered lists.
    """
    def convert_enumerate_block(match):
        items_content = match.group(1)
        items = re.findall(r'\\item\s+([^\\]*?)(?=\\item|\Z)', items_content, re.DOTALL)
        markdown_items = []
        for i, item in enumerate(items, 1):
            item = item.strip()
            if item:
                item = re.sub(r'\s+', ' ', item)
                markdown_items.append(f'{i}. {item}')
        return '\n'.join(markdown_items)
    
    content = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', 
                    convert_enumerate_block, content, flags=re.DOTALL)
    
    return content

def convert_sections(content):
    """
    Convert LaTeX section commands to markdown headers.
    Numbered sections get automatic counters, unnumbered sections don't.
    """
    # Section counters
    section_counter = 0
    subsection_counter = 0
    subsubsection_counter = 0
    
    def replace_section(match):
        nonlocal section_counter, subsection_counter, subsubsection_counter
        title = match.group(1)
        section_counter += 1
        subsection_counter = 0  # Reset subsection counter
        subsubsection_counter = 0  # Reset subsubsection counter
        return f'# {section_counter} {title}'
    
    def replace_subsection(match):
        nonlocal subsection_counter, subsubsection_counter
        title = match.group(1)
        subsection_counter += 1
        subsubsection_counter = 0  # Reset subsubsection counter
        return f'## {section_counter}.{subsection_counter} {title}'
    
    def replace_subsubsection(match):
        nonlocal subsubsection_counter
        title = match.group(1)
        subsubsection_counter += 1
        return f'### {section_counter}.{subsection_counter}.{subsubsection_counter} {title}'
    
    # Convert numbered sections
    content = re.sub(r'\\section\{([^}]+)\}', replace_section, content)
    content = re.sub(r'\\subsection\{([^}]+)\}', replace_subsection, content)
    content = re.sub(r'\\subsubsection\{([^}]+)\}', replace_subsubsection, content)
    
    # Convert unnumbered sections (starred versions)
    content = re.sub(r'\\section\*\{([^}]+)\}', r'# \1', content)
    content = re.sub(r'\\subsection\*\{([^}]+)\}', r'## \1', content)
    content = re.sub(r'\\subsubsection\*\{([^}]+)\}', r'### \1', content)
    
    return content

def convert_footnotes(content):
    """
    Convert LaTeX footnotes to markdown footnotes.
    Handles:
    - \footnote{text} -> [^n] with [^n]: text at end
    - \footnotemark -> [^n] (placeholder, needs manual \footnotetext)
    - \footnotetext{text} -> [^n]: text definition
    - \footnotemark[num] and \footnotetext[num]{text} -> [^num] and [^num]: text
    """
    footnotes = {}
    footnote_counter = 1
    
    # First pass: collect \footnotetext entries (both numbered and unnumbered)
    def collect_footnotetext(match):
        nonlocal footnote_counter
        number = match.group(1)  # Could be None for unnumbered
        text = match.group(2)
        
        if number:
            # Numbered footnotetext
            footnote_num = int(number)
            footnotes[footnote_num] = text
            # Update counter to be at least this high
            footnote_counter = max(footnote_counter, footnote_num + 1)
        else:
            # Unnumbered footnotetext - use next available number
            footnotes[footnote_counter] = text
            footnote_counter += 1
        
        return ''  # Remove the \footnotetext from content
    
    # Handle both \footnotetext[num]{text} and \footnotetext{text}
    content = re.sub(r'\\footnotetext(?:\[(\d+)\])?\{([^}]+)\}', collect_footnotetext, content)
    
    # Second pass: replace \footnotemark entries
    def replace_footnotemark(match):
        nonlocal footnote_counter
        number = match.group(1)  # Could be None for unnumbered
        
        if number:
            # Numbered footnotemark
            footnote_num = int(number)
            return f"[^{footnote_num}]"
        else:
            # Unnumbered footnotemark - use next available number
            result = f"[^{footnote_counter}]"
            footnote_counter += 1
            return result
    
    # Handle both \footnotemark[num] and \footnotemark
    content = re.sub(r'\\footnotemark(?:\[(\d+)\])?', replace_footnotemark, content)
    
    # Third pass: replace regular \footnote{text} entries
    def replace_footnote(match):
        nonlocal footnote_counter
        footnote_text = match.group(1)
        footnote_ref = f"[^{footnote_counter}]"
        footnotes[footnote_counter] = footnote_text
        footnote_counter += 1
        return footnote_ref
    
    content = re.sub(r'\\footnote\{([^}]+)\}', replace_footnote, content)
    
    # Add footnote definitions at the end if any footnotes were collected
    if footnotes:
        footnote_defs = []
        for num in sorted(footnotes.keys()):
            footnote_defs.append(f"[^{num}]: {footnotes[num]}")
        content += '\n\n' + '\n'.join(footnote_defs)
    
    return content

def convert_quotes(content):
    """
    Convert LaTeX quote environment to markdown blockquotes.
    """
    def convert_quote_block(match):
        quote_content = match.group(1).strip()
        # Split into lines and add > prefix
        lines = quote_content.split('\n')
        quoted_lines = ['> ' + line.strip() for line in lines if line.strip()]
        return '\n'.join(quoted_lines)
    
    content = re.sub(r'\\begin\{quote\}(.*?)\\end\{quote\}', 
                    convert_quote_block, content, flags=re.DOTALL)
    
    return content

def tex_to_ipynb(input_file, output_file=None):
    """
    Convert a .tex file to .ipynb file.
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    if not input_path.suffix.lower() == '.tex':
        raise ValueError("Input file must have .tex extension")
    
    # Determine output file
    if output_file is None:
        output_file = input_path.with_suffix('.ipynb')
    else:
        output_file = Path(output_file)
    
    # Read LaTeX content
    with open(input_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()
    
    # Convert to notebook
    notebook = convert_tex_to_ipynb(tex_content)
    
    # Write notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Converted {input_file} to {output_file}")
    return output_file

def main():
    """
    Command line interface for the converter.
    """
    parser = argparse.ArgumentParser(
        description='Convert LaTeX (.tex) files to Jupyter Notebook (.ipynb) format',
        epilog='Example: python tex_to_ipynb.py document.tex -o notebook.ipynb'
    )
    
    parser.add_argument(
        'input_file',
        help='Input LaTeX file (.tex)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output Jupyter notebook file (.ipynb). If not specified, uses input filename with .ipynb extension',
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        tex_to_ipynb(args.input_file, args.output)
        
    except FileNotFoundError as e:
        parser.error(f"File not found: {e}")
    except ValueError as e:
        parser.error(str(e))
    except Exception as e:
        parser.error(f"Conversion failed: {e}")

if __name__ == "__main__":
    main()