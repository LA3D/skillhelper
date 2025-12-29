def count_lines(filepath: str) -> dict:
    '''Count lines in a file'''
    with open(filepath) as f:
        lines = f.readlines()
    return {
        'total': len(lines),
        'non_empty': len([l for l in lines if l.strip()])
    }

def check_style(code: str) -> str:
    '''Check if code follows basic style rules'''
    issues = []
    if '  ' in code and '    ' not in code:
        issues.append('Uses 2-space indentation instead of 4')
    if len(code.split('\n')) > 100:
        issues.append('File is longer than 100 lines')
    return '\n'.join(issues) if issues else 'No style issues found'
