package cmd

import (
	"context"
	"strings"
	"testing"
)

func TestNewAuthManagerRegistersClineAndKilocode(t *testing.T) {
	t.Parallel()

	manager := newAuthManager()
	providers := []string{"cline", "kilocode"}

	for _, provider := range providers {
		provider := provider
		t.Run(provider, func(t *testing.T) {
			t.Parallel()

			_, _, err := manager.Login(context.Background(), provider, nil, nil)
			if err == nil {
				t.Fatalf("expected an error for %q with nil config", provider)
			}
			if strings.Contains(err.Error(), "not registered") {
				t.Fatalf("provider %q should be registered, got error: %v", provider, err)
			}
		})
	}
}
