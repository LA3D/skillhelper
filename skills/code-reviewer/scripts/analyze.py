#!/usr/bin/env python3
"""Simple code analysis script."""

def analyze_code(code: str) -> dict:
    """Analyze code and return basic metrics."""
    lines = code.split('\n')
    return {
        'lines': len(lines),
        'non_empty': len([l for l in lines if l.strip()]),
        'comments': len([l for l in lines if l.strip().startswith('#')])
    }

if __name__ == '__main__':
    import sys
    code = sys.stdin.read()
    print(analyze_code(code))
