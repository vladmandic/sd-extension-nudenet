# sdnext installer
"""
import installer

dependencies = ['onnxruntime']
for dependency in dependencies:
    if not installer.installed(dependency, reload=False, quiet=True):
        installer.install(dependency, ignore=False)
"""

# a1111 installer
"""
import launch

for dep in ['onnxruntime']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep}")
"""
