@echo off
echo ======================================================================
echo    SHA-512 Medical Image Encryption - Generating Encrypted Images
echo ======================================================================
echo.
cd /d "s:\NIIT\Sem 6\R&D\RnD"
.venv\Scripts\python.exe generate_encrypted_images.py
echo.
echo ======================================================================
echo                         Process Complete!
echo ======================================================================
echo.
echo Check the folder for:
echo   - encrypted_*.jpg/jpeg (encrypted versions - will look like noise)
echo   - decrypted_*.jpg/jpeg (decrypted versions - identical to originals)
echo.
pause
