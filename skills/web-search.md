---
name: web-search
description: Search the web using curl and a search API
tools: [shell]
---

Use the shell tool to search the web. Example:
```
curl -s "https://api.duckduckgo.com/?q={input}&format=json" | python3 -c "import sys,json; data=json.load(sys.stdin); [print(r.get('Text','')) for r in data.get('RelatedTopics',[])]"
```

Parse the results and summarize the findings for the user.
