/*
 * ============================================================================
 * TM4C123GH6PM Firmware: UART1 Receiver + Dynamic RGB LED Status Indicator
 * ============================================================================
 * Status Behavior:
 *   🟢 GREEN LED (PF3) = While recording audio (4.0s recording window on Pi)
 *   🔵 BLUE  LED (PF2) = Sound event found / classified (Street Music, Siren, etc.)
 *   🔴 RED   LED (PF1) = Script closed / stopped, idle background, or timeout
 *
 * Connections:
 *   Raspberry Pi Pin 8 (TXD)  ---> TM4C PB0 (Header J1 Pin 3)
 *   Raspberry Pi Pin 6 (GND)  ---> TM4C GND (Header J2 Pin 1)
 * ============================================================================
 */

#include "TM4C123GH6PM.h"
#include <stdint.h>
#include <string.h>

#define RX_BUFFER_SIZE      32
#define TIMEOUT_THRESHOLD   6000000  /* ~5-6 seconds timeout counter at 16 MHz */

/* Onboard RGB LED bit masks on GPIO Port F */
#define LED_OFF         0x00
#define LED_RED         0x02  /* PF1 = Red LED   (Stopped / Searching) */
#define LED_BLUE        0x04  /* PF2 = Blue LED  (Sound Found!)        */
#define LED_GREEN       0x08  /* PF3 = Green LED (Recording Mic)       */

/* ----------------------------------------------------------------------------
 * Initialize Onboard RGB LEDs (Port F: PF1, PF2, PF3)
 * ---------------------------------------------------------------------------- */
void LED_Init(void) {
    /* 1. Enable clock for GPIO Port F */
    SYSCTL->RCGCGPIO |= (1 << 5);              /* Bit 5 = Port F */
    while ((SYSCTL->PRGPIO & (1 << 5)) == 0);  /* Wait until ready */

    /* 2. Configure PF1 (Red), PF2 (Blue), and PF3 (Green) as digital outputs */
    GPIOF->DIR |= (LED_RED | LED_BLUE | LED_GREEN);
    GPIOF->DEN |= (LED_RED | LED_BLUE | LED_GREEN);

    /* 3. Initial power-on state: RED LED ON (Waiting for Pi) */
    GPIOF->DATA = (GPIOF->DATA & ~0x0E) | LED_RED;
}

void Set_LED(uint8_t color) {
    /* Clear PF1, PF2, PF3 and write selected color */
    GPIOF->DATA = (GPIOF->DATA & ~0x0E) | color;
}

/* ----------------------------------------------------------------------------
 * Initialize UART1 (PB0: U1Rx @ 9600 Baud)
 * Locked to 16.0 MHz PIOSC (UARTCC = 0x5) for exact baud rate synchronization.
 * ---------------------------------------------------------------------------- */
void UART1_Init(void) {
    /* 1. Enable clock for UART1 and GPIO Port B */
    SYSCTL->RCGCUART |= (1 << 1);              /* Bit 1 = UART1 */
    SYSCTL->RCGCGPIO |= (1 << 1);              /* Bit 1 = Port B */
    while ((SYSCTL->PRGPIO & (1 << 1)) == 0);

    /* 2. Configure PB0 as UART1 RX */
    GPIOB->AFSEL |= 0x01;                      /* Enable alt function on PB0 */
    GPIOB->PCTL   = (GPIOB->PCTL & 0xFFFFFFF0) | 0x00000001; /* Assign PB0 to U1Rx */
    GPIOB->DEN   |= 0x01;                      /* Digital enable PB0 */

    /* 3. Configure 9600 Baud rate @ 16 MHz PIOSC */
    UART1->CTL  &= ~(1 << 0);                  /* Disable UART during configuration */
    UART1->CC    = 0x5;                        /* PIOSC = strictly fixed 16 MHz clock */
    UART1->IBRD  = 104;                        /* 16,000,000 / (16 * 9600) = 104 */
    UART1->FBRD  = 11;                         /* round(0.166667 * 64) = 11 */
    UART1->LCRH  = 0x70;                       /* 8 bits, FIFOs enabled, no parity, 1 stop */
    UART1->CTL  |= 0x301;                      /* Enable UART1, TXE, RXE */
}

uint8_t UART1_IsAvailable(void) {
    /* RXFE (Receive FIFO Empty) bit 4: 0 = data available in FIFO */
    return ((UART1->FR & (1 << 4)) == 0);
}

char UART1_ReadChar(void) {
    while (UART1->FR & (1 << 4));              /* Wait while FIFO empty */
    return (char)(UART1->DR & 0xFF);
}

/* ----------------------------------------------------------------------------
 * Main Application
 * ---------------------------------------------------------------------------- */
int main(void) {
    char rx_buffer[RX_BUFFER_SIZE];
    uint8_t rx_idx = 0;
    char ch;
    uint32_t timeout_counter = 0;

    /* Initialize Onboard LEDs and UART1 */
    LED_Init();
    UART1_Init();

    /* Start in RED state (Stopped / Waiting) */
    Set_LED(LED_RED);

    while (1) {
        if (UART1_IsAvailable()) {
            ch = UART1_ReadChar();
            timeout_counter = 0; /* Reset timeout watchdog on incoming data */

            /* Check for End-of-Line delimiter (\n or \r) */
            if (ch == '\n' || ch == '\r') {
                if (rx_idx > 0) {
                    rx_buffer[rx_idx] = '\0';

                    /* 
                     * 1. While recording -> 🟢 GREEN LED
                     */
                    if (strstr(rx_buffer, "RECORDING") != NULL) {
                        Set_LED(LED_GREEN);
                    }
                    /* 
                     * 2. Script stopped / closed or quiet background -> 🔴 RED LED
                     */
                    else if (strstr(rx_buffer, "STOPPED") != NULL ||
                             strstr(rx_buffer, "DISCONNECTED") != NULL ||
                             strstr(rx_buffer, "Background Noise") != NULL ||
                             strstr(rx_buffer, "background_noise") != NULL) {
                        Set_LED(LED_RED);
                    }
                    /* 
                     * 3. Target sound found (Street Music, Siren, Drilling, etc.) -> 🔵 BLUE LED
                     */
                    else if (rx_idx >= 3) {
                        Set_LED(LED_BLUE);
                    }

                    rx_idx = 0; /* Reset buffer index */
                }
            }
            else if (rx_idx < (RX_BUFFER_SIZE - 1)) {
                /* Store printable characters */
                if (ch >= 32 && ch <= 126) {
                    rx_buffer[rx_idx++] = ch;
                }
            }
        } else {
            /* 
             * Automatic Heartbeat Timeout:
             * If Pi script stops or is closed and no messages arrive for ~5 seconds,
             * automatically switch back to 🔴 RED LED!
             */
            if (timeout_counter < TIMEOUT_THRESHOLD) {
                timeout_counter++;
            } else if (timeout_counter == TIMEOUT_THRESHOLD) {
                Set_LED(LED_RED); /* Turn RED if Pi disconnected */
                timeout_counter++; /* Prevent repeated trigger */
            }
        }
    }
}
