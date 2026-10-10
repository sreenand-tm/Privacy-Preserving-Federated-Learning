/*
 * lcd.h - 16x2 HD44780 LCD Driver for TM4C123GH6PM (4-Bit Mode)
 *
 * Pin Connections:
 *   LCD RS   -> PA5
 *   LCD EN   -> PA6
 *   LCD R/W  -> GND
 *   LCD D4   -> PB4
 *   LCD D5   -> PB5
 *   LCD D6   -> PB6
 *   LCD D7   -> PB7
 *   LCD VSS  -> GND
 *   LCD VDD  -> 5V
 *   LCD V0   -> 10k Potentiometer (Contrast) or 1k to GND
 */

#ifndef LCD_H_
#define LCD_H_

#include <stdint.h>

// Initialize LCD in 4-bit mode
void LCD_Init(void);

// Send a command to LCD (e.g. clear, set cursor)
void LCD_Command(uint8_t cmd);

// Send a single character to LCD
void LCD_Data(uint8_t data);

// Print a null-terminated string to LCD
void LCD_Print(const char* str);

// Set cursor to (row, col) - row: 0 or 1, col: 0 to 15
void LCD_SetCursor(uint8_t row, uint8_t col);

// Clear LCD display
void LCD_Clear(void);

// Microsecond delay helper
void LCD_Delay_Us(uint32_t us);

// Millisecond delay helper
void LCD_Delay_Ms(uint32_t ms);

#endif /* LCD_H_ */
