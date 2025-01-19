import nbformat
import os

def convert_notebook_to_py(notebook_path, output_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create Python file content
    py_content = "# Converted from notebook: " + os.path.basename(notebook_path) + "\n\n"
    
    # Process each cell
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Add code cells
            py_content += "# Cell\n"
            py_content += cell.source + "\n\n"
        elif cell.cell_type == 'markdown':
            # Add markdown cells as comments
            py_content += "# Markdown:\n"
            py_content += "# " + cell.source.replace('\n', '\n# ') + "\n\n"
    
    # Write to Python file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(py_content)

# Convert your notebook
notebook_path = 'i:/NSU/CV/fmd-project/model/mainnb.ipynb'
output_path = 'i:/NSU/CV/fmd-project/model/notebook_converted.py'
convert_notebook_to_py(notebook_path, output_path)