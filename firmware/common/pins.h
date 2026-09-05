/*
 * Shared, board-agnostic pin definitions.
 *
 * Include this in both the ESP32 and UNO Q MCU sketches, then override the
 * values below for the final wiring.
 */

#ifndef ACTIVESTEP_PINS_H
#define ACTIVESTEP_PINS_H

// --- ESP32 DevKit V1 (30 pin) defaults ---
#define ESP32_MOTOR_1       25
#define ESP32_LASER_1       26
#define ESP32_SW_TRUE       32
#define ESP32_SW_FALSE      33
#define ESP32_STATUS_LED    2
#define ESP32_SHANK_SDA     21
#define ESP32_SHANK_SCL     22
#define ESP32_SHANK_INT     19

// --- UNO Q MCU (STM32U585) defaults ---
#define UNOQ_MOTOR_2        5
#define UNOQ_LASER_2        6
#define UNOQ_SW_OK          2
#define UNOQ_STATUS_LED     13
#define UNOQ_TRUNK_SDA      20
#define UNOQ_TRUNK_SCL      21

// --- Generic I2C addresses ---
#define MPU6050_ADDR        0x68

#endif
