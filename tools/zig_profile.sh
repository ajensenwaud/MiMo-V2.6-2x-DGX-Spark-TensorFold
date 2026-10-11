#!/usr/bin/env bash
# Round profile of the Zig server (TF_MIMO_PROFILE=1): two single-stream chat replies, then four streams at once.
# On gx10 with the Python serve stopped; restarts it at the end.
set -u
T=$HOME/mimo/recipe/tools
EXTRA_ENV="-e TF_MIMO_PROFILE=1" BIN=${BIN:-/mimo/zig-out-z7} CONTEXT=${CONTEXT:-65536} bash $T/zig_serve.sh || exit 1
req() {
  curl -s -m 600 localhost:8892/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":600,\"temperature\":0,\"ignore_eos\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);t=d['tensorfold'];print('tokens',d['usage']['completion_tokens'],'tok/s',round(t.get('tokens_per_second',0),1))"
}
echo "--- 1 stream"
req "Tell me about the history of the city of Sydney, its people and its harbour."
req "Explain how a bicycle stays upright, in plain English, with examples."
docker logs mimo-zig 2>&1 | grep "mimo profile"
echo "--- 4 streams"
for q in "Describe the water cycle for a ten year old." "Write a short story about a lighthouse keeper." "Explain inflation and interest rates simply." "What makes a good cup of coffee? Discuss."; do req "$q" & done; wait
docker logs mimo-zig 2>&1 | grep "mimo profile" | tail -3
bash $T/zig_serve.sh stop
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
