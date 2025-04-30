"""
Script para construir orchestrator.py a partir de módulos refactorizados
"""
import os
import re
import datetime

def combine_modules():
    """Combina todos los módulos en un solo archivo orchestrator.py"""
    output_file = "c:\\Users\\DOM\\Desktop\\content-bot\\brain\\orchestrator.py"
    module_dirs = [
        "c:\\Users\\DOM\\Desktop\\content-bot\\brain\\orchestrator\\utils",
        "c:\\Users\\DOM\\Desktop\\content-bot\\brain\\orchestrator\\core",
        "c:\\Users\\DOM\\Desktop\\content-bot\\brain\\orchestrator\\managers"
    ]
    
    # Guardar una copia de seguridad del archivo original
    if os.path.exists(output_file):
        backup_file = f"{output_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.rename(output_file, backup_file)
        print(f"Backup creado: {backup_file}")
    
    imports = set()
    code_blocks = []
    
    # Extraer docstring del archivo original si existe
    docstring = ""
    if os.path.exists(f"{output_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"):
        with open(f"{output_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}", "r", encoding="utf-8") as f:
            content = f.read()
            docstring_match = re.match(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = f'"""{docstring_match.group(1)}"""\n\n'
    
    # Recopilar todos los imports y código
    for module_dir in module_dirs:
        for root, _, files in os.walk(module_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Extraer imports
                        import_lines = re.findall(r'^(?:import|from) .*$', content, re.MULTILINE)
                        for line in import_lines:
                            imports.add(line)
                        
                        # Eliminar imports y docstrings del código
                        code = re.sub(r'^(?:import|from) .*$', '', content, flags=re.MULTILINE)
                        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
                        
                        if code.strip():
                            module_path = os.path.relpath(file_path, "c:\\Users\\DOM\\Desktop\\content-bot\\brain\\orchestrator")
                            code_blocks.append(f"# Módulo: {module_path}\n{code.strip()}\n\n")
    
    # Escribir el archivo combinado
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(docstring)
        f.write("# ARCHIVO GENERADO AUTOMÁTICAMENTE - NO EDITAR DIRECTAMENTE\n")
        f.write("# Editar los módulos individuales en la carpeta 'orchestrator'\n\n")
        
        # Escribir imports ordenados
        f.write("# Imports\n")
        for imp in sorted(imports):
            f.write(f"{imp}\n")
        f.write("\n\n")
        
        # Escribir bloques de código
        for block in code_blocks:
            f.write(block)
        
        # Añadir el bloque main
        f.write("""
if __name__ == "__main__":
    orchestrator = Orchestrator()
    try:
        orchestrator.start()
        while orchestrator.active:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()
""")
    
    print(f"Archivo combinado creado: {output_file}")

if __name__ == "__main__":
    combine_modules()