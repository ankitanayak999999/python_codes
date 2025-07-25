pyinstaller --onefile --console ^
--name Smart_Data_Scanner ^
--hidden-import=pyarmor ^
--hidden-import=pyarmor_runtime ^
dist/Smart_Data_Scanner.py
