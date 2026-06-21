class JobInput:
    def __init__(self, job):
        job = job or {}
        self.raw = dict(job or {})
        self.llm_input = job.get("messages", job.get("prompt"))
        self.stream = job.get("stream", False)
        self.openai_route = job.get("openai_route")
        self.openai_input = job.get("openai_input") or self._extract_openai_input()

    def _extract_openai_input(self):
        excluded = {"openai_route", "openai_input"}
        return {
            key: value
            for key, value in self.raw.items()
            if key not in excluded and value is not None
        }
