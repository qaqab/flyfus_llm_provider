# Flyfus LLM Error Protocol

Runtime model failures remain Dify errors. The error message contains one envelope:

```text
<FLYFUS_ERROR>{...valid JSON...}</FLYFUS_ERROR>
```

The JSON fields are:

- `type`: stable root-cause classification.
- `user_message`: safe text for end users.
- `retryable`: whether an outer caller may submit the whole task again.
- `partial_output`: whether any visible or reasoning chunk reached Dify before failure.
- `log_id`, `invocation_id`, `request_id`, `client_request_id`: the plugin invocation identifier.
- `response_id`: the model response identifier, when the upstream produced one.
- `upstream_request_id`, `cf_ray`: optional identifiers returned by the upstream service.
- `error`: complete safe diagnostics. Traceback, replay body, prompts, and credentials remain in SLS.

The envelope intentionally has no protocol-version field. Existing fields must remain backward compatible;
new fields may only be added.

## Retry and classification rules

| Type | Trigger | Internal retry | Outer retryable |
|---|---|---:|---:|
| `upstream_stream_incomplete` | GPT/Responses stream disconnects or ends without `response.completed` | Once, only before any chunk | Yes |
| `upstream_response_incomplete` | GPT explicitly returns `response.incomplete` | None | No |
| `upstream_response_failed` | GPT explicitly returns an unclassified `response.failed` error | None | Yes |
| `upstream_timeout` | Request connect/read timeout or a failed response identifying a timeout | Once | Yes |
| `upstream_connection_error` | Request connection failure | Once | Yes |
| `upstream_rate_limited` | HTTP 429 or a failed response identifying rate limiting | Once | Yes |
| `upstream_server_error` | HTTP 5xx or a failed response identifying a server/overload error | Once | Yes |
| `upstream_auth_error` | HTTP 401/403 or an authentication failure | None | No |
| `upstream_bad_request` | HTTP 4xx request/parameter/model error other than 408/429 | None | No |
| `gemini_stream_incomplete` | Gemini stream ends without `finishReason` | Once, only before any chunk | Yes |
| `gemini_empty_response` | Gemini finishes without visible text or a tool call | Twice, only before any chunk | No after exhaustion |
| `gemini_malformed_function_call` | Gemini returns `MALFORMED_FUNCTION_CALL` | Twice, only before any chunk | No after exhaustion |
| `gemini_partial_output_error` | Any Gemini error after a chunk reached Dify | None | No |
| `gemini_safety_blocked` | Gemini prompt/candidate is blocked by Safety | None | No |
| `context_window_exceeded` | Mandatory context remains too large after emergency trimming | One trimmed-context retry | No |
| `invalid_message_protocol` | Tool-call history is orphaned, incomplete, or duplicated | None | No |
| `model_error` | Unclassified runtime failure | None | No |

Two retries means at most three upstream attempts: the initial request plus two retries.
For HTTP 429, the request retry respects a numeric `Retry-After` header, defaults to one second, and is
capped at ten seconds.

`retry_count` in the diagnostic text counts model request retries recorded as `request_retry`,
`stream_retry`, `gemini_retry`, or `context_guard_retry`.

## Partial output

The plugin cannot retract chunks already sent to Dify. When `partial_output` is `true`, a client should not
present those chunks as a complete answer. It may hide them or label them as incomplete, then display
`user_message`.

## Parsing

Dify may wrap the envelope inside its own serialized `description` and `message` fields. Consumers should
search the complete error string for the first `<FLYFUS_ERROR>...</FLYFUS_ERROR>` block, parse the enclosed
JSON, and deduplicate repeated wrappers by `log_id`.
