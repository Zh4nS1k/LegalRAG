import ast
import json

def get_structure(filepath):
    with open(filepath, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    structure = []
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append({
                        'name': child.name,
                        'docstring': ast.get_docstring(child) or 'No docstring',
                        'lineno': child.lineno
                    })
            structure.append({
                'type': 'class',
                'name': node.name,
                'docstring': ast.get_docstring(node) or 'No docstring',
                'lineno': node.lineno,
                'methods': methods
            })
        elif isinstance(node, ast.FunctionDef):
            structure.append({
                'type': 'function',
                'name': node.name,
                'docstring': ast.get_docstring(node) or 'No docstring',
                'lineno': node.lineno
            })
            
    return structure

if __name__ == '__main__':
    struct = get_structure('ai_service/retrieval/rag_chain.py')
    print(json.dumps(struct, indent=2))
