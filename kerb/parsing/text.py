"""Text extraction utilities.

This module provides functions for extracting structured content from text,
including XML tags, markdown sections, lists, and tables.
"""

import re
from typing import Dict, List


def extract_xml_tag(text: str, tag: str) -> List[str]:
    """Extract content from XML-style tags.
    
    Args:
        text (str): Text containing XML tags
        tag (str): Tag name to extract (without < >)
        
    Returns:
        List[str]: List of tag contents
        
    Examples:
        >>> extract_xml_tag('<answer>42</answer>', 'answer')
        ['42']
    """
    pattern = f'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_markdown_sections(text: str, heading_level: int = 2) -> Dict[str, str]:
    """Extract sections from markdown by heading level.
    
    Args:
        text (str): Markdown text
        heading_level (int): Heading level to split on (1-6)
        
    Returns:
        Dict[str, str]: Mapping of heading names to section content
    """
    heading_pattern = f'^{"#" * heading_level}\\s+(.+?)$'
    sections = {}
    
    lines = text.split('\n')
    current_heading = None
    current_content = []
    
    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            # Save previous section
            if current_heading:
                sections[current_heading] = '\n'.join(current_content).strip()
            
            # Start new section
            current_heading = match.group(1).strip()
            current_content = []
        else:
            if current_heading:
                current_content.append(line)
    
    # Save last section
    if current_heading:
        sections[current_heading] = '\n'.join(current_content).strip()
    
    return sections


def extract_list_items(text: str, ordered: bool = False) -> List[str]:
    """Extract list items from markdown text.
    
    Args:
        text (str): Markdown text
        ordered (bool): Extract ordered lists (1. 2. 3.) vs unordered (- * +)
        
    Returns:
        List[str]: List items
    """
    if ordered:
        pattern = r'^\d+\.\s+(.+)$'
    else:
        pattern = r'^[-*+]\s+(.+)$'
    
    items = []
    for line in text.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            items.append(match.group(1))
    
    return items


def parse_markdown_table(text: str) -> List[Dict[str, str]]:
    """Parse a markdown table into a list of dictionaries.
    
    Args:
        text (str): Markdown table text
        
    Returns:
        List[Dict[str, str]]: List of rows as dictionaries
        
    Examples:
        >>> table = '''
        ... | Name | Age |
        ... |------|-----|
        ... | John | 30  |
        ... | Jane | 25  |
        ... '''
        >>> parse_markdown_table(table)
        [{'Name': 'John', 'Age': '30'}, {'Name': 'Jane', 'Age': '25'}]
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if len(lines) < 2:
        return []
    
    # Parse header
    header = [col.strip() for col in lines[0].split('|') if col.strip()]
    
    # Skip separator line (line with dashes)
    data_lines = [line for line in lines[2:] if not re.match(r'^[\s|:-]+$', line)]
    
    # Parse rows
    rows = []
    for line in data_lines:
        # Split by | and get all parts
        parts = line.split('|')
        # Filter out empty strings but keep track of positions
        values = [val.strip() for val in parts]
        # Remove leading/trailing empty strings from pipe at start/end
        if values and values[0] == '':
            values = values[1:]
        if values and values[-1] == '':
            values = values[:-1]
        
        if len(values) == len(header):
            rows.append(dict(zip(header, values)))
    
    return rows
