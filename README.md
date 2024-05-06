# study_lanchain

## 설치 및 테스트 방법
`Hugging Face와 ollama를 통해 설치 방법은 다음과 같다.`

1. gguf 모델 다운로드 

gguf 모델 다운로드
https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/tree/main  



Meta-Llama-3-8B-Instruct.Q5_1.gguf



2. Modelfile 생성
```
FROM Meta-Llama-3-8B-Instruct.Q8_0.gguf

TEMPLATE """{{- if .System }}
<|begin_of_text|>system {{ .System }}<|end_of_text|>
{{- end }}
<|begin_of_text|>user
{{ .Prompt }}<|end_of_text|>
<|begin_of_text|>assistant
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop <|begin_of_text|>
PARAMETER stop <|end_of_text|>
PARAMETER stop <|eot_id|>
PARAMETER stop <|end_of_text|>
```

3. Ollama 모델 생성 및 확인

ollama create llama3-instruct-8b -f Modelfile

ollama list

4. Ollama 실행

ollama run llama3-instruct-8b

설치 사용 리뷰 

일단 한글은 파인튜닝되지 않아, 정확하지 않았으나,

영어로 물어본 것에 대해서는 8B 모델이라도 매우 훌륭하다. (php오류를 찾아 물어봤더니 굉장히 정확히 찾았다)



WebUI 설치 방법

Assuming you already have Docker and Ollama running on your computer, installation is super simple.

docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

The simply go to http://localhost:3000, make an account, and start chatting away!
