import os
import shutil


for folder in os.listdir("doc/build"):
    shutil.rmtree("doc/build/%s" % folder)
for file in os.listdir("doc/source"):
    if file.endswith(".rst") and file!="index.rst":
        os.remove("doc/source/%s" % file)
os.system("sphinx-apidoc -o doc/source src")
os.chdir("doc")
os.system("make html")
