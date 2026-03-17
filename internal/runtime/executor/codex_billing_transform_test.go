package executor

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// ---------------------------------------------------------------------------
// codexMergeToolOutputBlocks tests
// ---------------------------------------------------------------------------

func TestCodexMergeToolOutputBlocks_NoInput(t *testing.T) {
	t.Parallel()
	body := []byte(`{"model":"gpt-5-codex"}`)
	got := codexMergeToolOutputBlocks(body)
	if string(got) != string(body) {
		t.Fatalf("expected no change, got %s", string(got))
	}
}

func TestCodexMergeToolOutputBlocks_NoFunctionCallOutput(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`)
	got := codexMergeToolOutputBlocks(body)
	if string(got) != string(body) {
		t.Fatalf("expected no change, got %s", string(got))
	}
}

func TestCodexMergeToolOutputBlocks_MergesFollowingUserMessage(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"function_call_output","call_id":"c1","output":"tool result"},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}` +
		`]}`)
	got := codexMergeToolOutputBlocks(body)

	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 1 {
		t.Fatalf("input len = %d, want 1 (user message should be merged)", len(items))
	}
	output := items[0].Get("output").String()
	if !strings.Contains(output, "tool result") {
		t.Fatalf("output should contain original text, got %q", output)
	}
	if !strings.Contains(output, "Please execute skill now:continue") {
		t.Fatalf("output should contain merged text, got %q", output)
	}
}

func TestCodexMergeToolOutputBlocks_MultipleTextItems(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"function_call_output","call_id":"c1","output":"A"},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"B"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"C"}]}` +
		`]}`)
	got := codexMergeToolOutputBlocks(body)

	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 1 {
		t.Fatalf("input len = %d, want 1", len(items))
	}
	output := items[0].Get("output").String()
	if !strings.Contains(output, "B") || !strings.Contains(output, "C") {
		t.Fatalf("output should contain both B and C, got %q", output)
	}
}

func TestCodexMergeToolOutputBlocks_NonUserItemNotMerged(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"function_call_output","call_id":"c1","output":"A"},` +
		`{"type":"function_call","call_id":"c2","name":"read","arguments":"{}"}` +
		`]}`)
	got := codexMergeToolOutputBlocks(body)

	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 2 {
		t.Fatalf("input len = %d, want 2 (function_call should not be merged)", len(items))
	}
}

func TestCodexMergeToolOutputBlocks_StringContentUserMessage(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"function_call_output","call_id":"c1","output":"A"},` +
		`{"type":"message","role":"user","content":"plain text"}` +
		`]}`)
	got := codexMergeToolOutputBlocks(body)

	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 1 {
		t.Fatalf("input len = %d, want 1", len(items))
	}
	output := items[0].Get("output").String()
	if !strings.Contains(output, "plain text") {
		t.Fatalf("output should contain plain text, got %q", output)
	}
}

// ---------------------------------------------------------------------------
// codexTransformUserToFunctionOutput tests
// ---------------------------------------------------------------------------

func TestCodexTransformUserToFunctionOutput_NoInput(t *testing.T) {
	t.Parallel()
	body := []byte(`{"model":"gpt-5-codex"}`)
	got := codexTransformUserToFunctionOutput(body)
	if string(got) != string(body) {
		t.Fatalf("expected no change, got %s", string(got))
	}
}

func TestCodexTransformUserToFunctionOutput_OnlyUserNoAssistant(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	got := codexTransformUserToFunctionOutput(body)
	// No assistant → no transformation.
	if string(got) != string(body) {
		t.Fatalf("expected no change, got %s", string(got))
	}
}

func TestCodexTransformUserToFunctionOutput_FollowUpUserBecomesToolPair(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"followup"}]}` +
		`]}`)
	got := codexTransformUserToFunctionOutput(body)

	items := gjson.GetBytes(got, "input").Array()
	// user(1) + assistant(1) + function_call(injected) + function_call_output(converted) = 4
	if len(items) != 4 {
		t.Fatalf("input len = %d, want 4; got %s", len(items), gjson.GetBytes(got, "input").Raw)
	}
	if items[0].Get("role").String() != "user" {
		t.Fatal("first item should remain user message")
	}
	if items[1].Get("role").String() != "assistant" {
		t.Fatal("second item should remain assistant message")
	}
	if items[2].Get("type").String() != "function_call" {
		t.Fatalf("third item should be function_call, got %q", items[2].Get("type").String())
	}
	if items[2].Get("name").String() != "ask_user" {
		t.Fatalf("function_call name should be ask_user, got %q", items[2].Get("name").String())
	}
	callID := items[2].Get("call_id").String()
	if callID == "" {
		t.Fatal("function_call call_id should not be empty")
	}
	if items[3].Get("type").String() != "function_call_output" {
		t.Fatalf("fourth item should be function_call_output, got %q", items[3].Get("type").String())
	}
	if items[3].Get("call_id").String() != callID {
		t.Fatal("function_call_output call_id should match function_call call_id")
	}
	if !strings.Contains(items[3].Get("output").String(), "followup") {
		t.Fatalf("function_call_output.output should contain followup text, got %q", items[3].Get("output").String())
	}
}

func TestCodexTransformUserToFunctionOutput_AlreadyFunctionCallOutputNotTransformed(t *testing.T) {
	t.Parallel()
	// Real tool result — should not be wrapped again.
	body := []byte(`{"input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will help"}]},` +
		`{"type":"function_call","call_id":"c1","name":"read","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"c1","output":"file content"}` +
		`]}`)
	got := codexTransformUserToFunctionOutput(body)

	// Should be unchanged: no plain user message after assistant without function_call.
	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 4 {
		t.Fatalf("input len = %d, want 4 (no extra injection); got %s", len(items), gjson.GetBytes(got, "input").Raw)
	}
}

func TestCodexTransformUserToFunctionOutput_StringContent(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"message","role":"user","content":"first"},` +
		`{"type":"message","role":"assistant","content":"response"},` +
		`{"type":"message","role":"user","content":"second"}` +
		`]}`)
	got := codexTransformUserToFunctionOutput(body)

	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 4 {
		t.Fatalf("input len = %d, want 4; got %s", len(items), gjson.GetBytes(got, "input").Raw)
	}
	if items[3].Get("output").String() != "second" {
		t.Fatalf("output = %q, want %q", items[3].Get("output").String(), "second")
	}
}

func TestCodexTransformUserToFunctionOutput_MixedContentTextExtracted(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"describe"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"sure"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"explain this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]}` +
		`]}`)
	got := codexTransformUserToFunctionOutput(body)

	items := gjson.GetBytes(got, "input").Array()
	if len(items) != 4 {
		t.Fatalf("input len = %d, want 4; got %s", len(items), gjson.GetBytes(got, "input").Raw)
	}
	output := items[3].Get("output").String()
	if !strings.Contains(output, "explain this") {
		t.Fatalf("output should contain text portion, got %q", output)
	}
}
