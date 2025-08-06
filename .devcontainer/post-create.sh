
BIN_PATH=$HOME/.local/bin
echo 'npx @google/gemini-cli@latest $@' > $BIN_PATH/gemini
chmod +x $BIN_PATH/gemini
echo 'npx @anthropic-ai/claude-code@latest $@' > $BIN_PATH/claude
chmod +x $BIN_PATH/claude

pip3 install --user -r requirements.txt