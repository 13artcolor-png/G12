import os

# Extensions de fichiers à lire (ajoute les tiennes si besoin)
extensions = ['.py', '.js', '.html', '.css', '.java', '.cs', '.php', '.sql', '.md']
fichier_sortie = "code_complet.txt"

with open(fichier_sortie, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk("."):
        # Ignorer les dossiers lourds ou inutiles
        if '.git' in root or 'node_modules' in root or '__pycache__' in root or 'venv' in root:
            continue
            
        for file in files:
            if any(file.endswith(ext) for ext in extensions) and file != "extract.py":
                filepath = os.path.join(root, file)
                outfile.write(f"\n{'='*20}\nFICHIER: {filepath}\n{'='*20}\n")
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Erreur de lecture: {e}")

print(f"Tout le code a été compilé dans {fichier_sortie}. Tu peux copier son contenu !")