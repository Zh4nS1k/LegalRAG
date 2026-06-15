// string_utils.go

package utils

import "strings"

// ToLower converts a string to lowercase
func ToLower(s string) string {
	return strings.ToLower(s)
}

// Contains checks if a string contains a substring
func Contains(s, substr string) bool {
	return strings.Contains(s, substr)
}

// IndexOf finds the index of a substring in a string, returns -1 if not found
func IndexOf(s, substr string) int {
	return strings.Index(s, substr)
}

// Split splits a string by a separator
func Split(s, sep string) []string {
	return strings.Split(s, sep)
}

// Trim removes leading and trailing whitespace
func Trim(s string) string {
	return strings.TrimSpace(s)
}

// Replace replaces all occurrences of old with new
func Replace(s, old, new string) string {
	return strings.ReplaceAll(s, old, new)
}
