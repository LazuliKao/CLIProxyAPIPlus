package cmd

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	sdkAuth "github.com/router-for-me/CLIProxyAPI/v6/sdk/auth"
	log "github.com/sirupsen/logrus"
)

// DoCodeBuddyLogin triggers the browser OAuth polling flow for CodeBuddy and saves tokens.
// It initiates the OAuth authentication, displays the user code for the user to enter
// at the CodeBuddy verification URL, and waits for authorization before saving the tokens.
//
// Parameters:
//   - cfg: The application configuration containing proxy and auth directory settings
//   - options: Login options including browser behavior settings
func DoCodeBuddyLogin(cfg *config.Config, options *LoginOptions) {
	if options == nil {
		options = &LoginOptions{}
	}

	// 让用户选择登录站点
	var useGlobal bool
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("请选择 CodeBuddy 登录站点:")
	fmt.Println("  [1] 中国站点 (copilot.tencent.com) 默认")
	fmt.Println("  [2] 全球站点 (www.codebuddy.ai)")
	fmt.Print("输入选择 (回车使用默认): ")
	choice, _ := reader.ReadString('\n')
	choice = strings.TrimSpace(choice)
	if choice == "2" {
		useGlobal = true
		fmt.Println("已选择全球站点 (www.codebuddy.ai)")
	} else {
		fmt.Println("已选择中国站点 (copilot.tencent.com)")
	}

	manager := newAuthManager()
	authOpts := &sdkAuth.LoginOptions{
		NoBrowser: options.NoBrowser,
		Metadata: map[string]string{
			"useGlobal": fmt.Sprintf("%t", useGlobal),
		},
	}

	record, savedPath, err := manager.Login(context.Background(), "codebuddy", cfg, authOpts)
	if err != nil {
		log.Errorf("CodeBuddy authentication failed: %v", err)
		return
	}

	if savedPath != "" {
		fmt.Printf("Authentication saved to %s\n", savedPath)
	}
	if record != nil && record.Label != "" {
		fmt.Printf("Authenticated as %s\n", record.Label)
	}
	fmt.Println("CodeBuddy authentication successful!")
}
