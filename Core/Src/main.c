/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "network.h"
#include "network_data.h"
#include "stm32f4xx_hal.h"
#include "stm32469i_discovery.h"
#include "stm32469i_discovery_lcd.h"
#include "stm32469i_discovery_ts.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* === Globals for AI === */
static ai_handle network = AI_HANDLE_NULL;
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

static ai_buffer ai_input[AI_NETWORK_IN_NUM];
static ai_buffer ai_output[AI_NETWORK_OUT_NUM];

static float in_data[AI_NETWORK_IN_1_SIZE];
static float out_data[AI_NETWORK_OUT_1_SIZE];

/* === Capture parameters / LCD layout === */
#ifndef LCD_FB_START_ADDRESS
#define LCD_FB_START_ADDRESS  ((uint32_t)0xC0000000)
#endif

#define SRC_WIDTH   800
#define SRC_HEIGHT  480

/* Drawing box on screen */
#define DRAW_X  50
#define DRAW_Y  80
#define DRAW_W  700
#define DRAW_H  250

/* Result area */
#define RESULT_X 200
#define RESULT_Y 350
#define RESULT_W 400
#define RESULT_H 110

/* Invert input if you draw black on white */
#define INVERT_INPUT 1

/* Small padding around bounding box captured */
#define BBOX_PAD 8


/* === Prototypes === */
static void network_init(void);
static int ai_run(uint8_t *img28); /* returns predicted class 0..9, or -1 on error */
static int ai_run_with_conf(uint8_t *img28); /* wrapper to run and keep out_data */
static void capture_frame_bbox_to_mnist(uint8_t *out28x28, int x0, int y0, int x1, int y1);
static void draw_ui_init(void);
static void Error_Display(const char *msg);

/* === Initialize network === */
static void network_init(void) {
    ai_error err;

    err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        /* cannot create network: show error and stop */
        Error_Display("AI create error");
        Error_Handler();
    }

    const ai_network_params params = {
        AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        AI_NETWORK_DATA_ACTIVATIONS(activations)
    };

    if (!ai_network_init(network, &params)) {
        /* initialization failed: show error and stop */
        err = ai_network_get_error(network);
        Error_Display("AI init error");
        Error_Handler();
    }

    /* Retrieve io buffers */
    const ai_buffer *input = ai_network_inputs_get(network, NULL);
    const ai_buffer *output = ai_network_outputs_get(network, NULL);

    /* copy descriptors */
    ai_input[0] = input[0];
    ai_output[0] = output[0];

    /* associate memory pointers (we store floats in in_data/out_data) */
    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);
}

/* === AI inference: convert uint8 28x28 -> float normalized and run ===
   Returns predicted label 0..9 or -1 on error. Also fills out_data[] with outputs.
   IMPORTANT: If your model was quantized you must apply the proper scale/zero_point here.
*/
static int ai_run(uint8_t *img) {
    ai_i32 nbatch;

    /* Normalize into in_data (float32 in range 0..1) */
    for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
        in_data[i] = ((float)img[i]) / 255.0f;
    }

    /* ensure ai_input/ai_output point to buffers (redundant but safe) */
    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    nbatch = ai_network_run(network, ai_input, ai_output);
    if (nbatch != 1) {
        /* Show error but do not block entire system */
        Error_Display("AI run error");
        return -1;
    }

    /* Find top index */
    int max_idx = 0;
    float max_val = out_data[0];
    for (int i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
        if (out_data[i] > max_val) {
            max_val = out_data[i];
            max_idx = i;
        }
    }

    return max_idx;
}

/* Convenience wrapper (same as ai_run but just a named alias if needed) */
static int ai_run_with_conf(uint8_t *img28) {
    return ai_run(img28);
}

/* === Capture a bounding-box region of the framebuffer and downscale to 28x28.
   - x0,y0..x1,y1 are inclusive screen coordinates.
   - out28x28[] values 0..255 (uint8_t).
*/
static void capture_frame_bbox_to_mnist(uint8_t *out28x28, int x0, int y0, int x1, int y1) {
    uint32_t *fb = (uint32_t *)LCD_FB_START_ADDRESS;
    if (fb == NULL) {
        for (int i = 0; i < 28*28; i++) out28x28[i] = 0;
        return;
    }

    int srcW = x1 - x0 + 1;
    int srcH = y1 - y0 + 1;
    if (srcW <= 0 || srcH <= 0) {
        for (int i = 0; i < 28*28; i++) out28x28[i] = 0;
        return;
    }

    const int dstW = 28, dstH = 28;
    float sx = (float)srcW / (float)dstW;
    float sy = (float)srcH / (float)dstH;

    for (int y = 0; y < dstH; y++) {
        float y0f = y * sy;
        float y1f = (y + 1) * sy;
        int ys = (int)floorf(y0f);
        int ye = (int)ceilf(y1f);
        if (ye >= srcH) ye = srcH - 1;
        if (ys < 0) ys = 0;

        for (int x = 0; x < dstW; x++) {
            float x0f = x * sx;
            float x1f = (x + 1) * sx;
            int xs = (int)floorf(x0f);
            int xe = (int)ceilf(x1f);
            if (xe >= srcW) xe = srcW - 1;
            if (xs < 0) xs = 0;

            uint32_t rsum=0, gsum=0, bsum=0, count=0;
            for (int yy = ys; yy <= ye; yy++) {
                uint32_t *row = fb + ((y0 + yy) * SRC_WIDTH);
                for (int xx = xs; xx <= xe; xx++) {
                    uint32_t px = row[x0 + xx]; /* ARGB8888 */
                    uint8_t b = (uint8_t)(px & 0xFF);
                    uint8_t g = (uint8_t)((px >> 8) & 0xFF);
                    uint8_t r = (uint8_t)((px >> 16) & 0xFF);
                    rsum += r; gsum += g; bsum += b;
                    count++;
                }
            }
            if (count == 0) count = 1;
            uint8_t ravg = (uint8_t)(rsum / count);
            uint8_t gavg = (uint8_t)(gsum / count);
            uint8_t bavg = (uint8_t)(bsum / count);

            uint8_t gray = (uint8_t)((0.299f * ravg) + (0.587f * gavg) + (0.114f * bavg));
#if INVERT_INPUT
            gray = 255 - gray;
#endif
            out28x28[y*dstW + x] = gray;
        }
    }
}

/* Small helper to draw the initial UI: top banner, drawing rectangle, result box */
static void draw_ui_init(void) {
    /* Clear screen */
    BSP_LCD_Clear(LCD_COLOR_WHITE);

    /* Top banner */
    BSP_LCD_SetBackColor(LCD_COLOR_BLUE);
    BSP_LCD_SetTextColor(LCD_COLOR_WHITE);
    BSP_LCD_FillRect(0, 0, SRC_WIDTH, 50);
    BSP_LCD_DisplayStringAt(0, 15, (uint8_t*)"Digit Recognition", CENTER_MODE);

    /* Drawing box */
    BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
    BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
    BSP_LCD_DrawRect(DRAW_X, DRAW_Y, DRAW_W, DRAW_H);
    BSP_LCD_DisplayStringAt(0, DRAW_Y - 18, (uint8_t*)"Draw here", CENTER_MODE);

    /* Clear drawing area (white) */
    BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
    BSP_LCD_FillRect(DRAW_X+1, DRAW_Y+1, DRAW_W-2, DRAW_H-2);

    /* "Clear" button (visual only: we clear automatically after prediction) */
    BSP_LCD_SetBackColor(LCD_COLOR_LIGHTGRAY);
    BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
    BSP_LCD_FillRect(DRAW_X, DRAW_Y + DRAW_H + 12, 120, 36);
    BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
    BSP_LCD_DisplayStringAt(DRAW_X + 10, DRAW_Y + DRAW_H + 22, (uint8_t*)"Clear", LEFT_MODE);

    /* Result box */
    BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
    BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
    BSP_LCD_DrawRect(RESULT_X, RESULT_Y, RESULT_W, RESULT_H);
    BSP_LCD_DisplayStringAt(0, RESULT_Y - 14, (uint8_t*)"Result", CENTER_MODE);

    /* Initial instruction */
    BSP_LCD_SetTextColor(LCD_COLOR_RED);
    BSP_LCD_DisplayStringAt(0, 60, (uint8_t*)"Draw a digit and release to predict", CENTER_MODE);
}

/* Error display (non-blocking) */
static void Error_Display(const char *msg) {
    /* Draw message in result area in red */
    BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
    BSP_LCD_SetTextColor(LCD_COLOR_RED);
    BSP_LCD_FillRect(RESULT_X+1, RESULT_Y+1, RESULT_W-2, RESULT_H-2);
    BSP_LCD_DisplayStringAt(0, RESULT_Y + (RESULT_H/2) - 6, (uint8_t*)msg, CENTER_MODE);
}


/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

DMA2D_HandleTypeDef hdma2d;

DSI_HandleTypeDef hdsi;

I2C_HandleTypeDef hi2c1;

LTDC_HandleTypeDef hltdc;

SDRAM_HandleTypeDef hsdram1;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_CRC_Init(void);
static void MX_DMA2D_Init(void);
static void MX_DSIHOST_DSI_Init(void);
static void MX_FMC_Init(void);
static void MX_I2C1_Init(void);
static void MX_LTDC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */


  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_DMA2D_Init();
  MX_DSIHOST_DSI_Init();
  MX_FMC_Init();
  MX_I2C1_Init();
  MX_LTDC_Init();
  /* USER CODE BEGIN 2 */
  network_init();


  /* Init SDRAM + LCD + Touch */
      BSP_SDRAM_Init();
      BSP_LCD_Init();
      BSP_LCD_LayerDefaultInit(0, LCD_FB_START_ADDRESS);
      BSP_LCD_SelectLayer(0);
      BSP_LCD_DisplayOn();

      if (BSP_TS_Init(SRC_WIDTH, SRC_HEIGHT) != TS_OK) {
          Error_Display("Touchscreen init error");
          Error_Handler();
      }

      /* draw UI */
      draw_ui_init();

      /* Touch bounding box tracking */
      int draw_min_x = SRC_WIDTH, draw_min_y = SRC_HEIGHT, draw_max_x = 0, draw_max_y = 0;
      int was_touching = 0;

      /* Buffers */
      uint8_t mnist_img[28*28];
      int prediction = -1;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
	  TS_StateTypeDef TS_State;
	          BSP_TS_GetState(&TS_State);

	          if (TS_State.touchDetected > 0) {
	              /* User is touching: update bounding box and draw pixels */
	              was_touching = 1;
	              for (int i = 0; i < TS_State.touchDetected; i++) {
	                  int tx = TS_State.touchX[i];
	                  int ty = TS_State.touchY[i];

	                  /* Constrain to drawing box so we don't mark UI */
	                  if (tx < DRAW_X) tx = DRAW_X;
	                  if (tx >= DRAW_X + DRAW_W) tx = DRAW_X + DRAW_W - 1;
	                  if (ty < DRAW_Y) ty = DRAW_Y;
	                  if (ty >= DRAW_Y + DRAW_H) ty = DRAW_Y + DRAW_H - 1;

	                  /* update bbox relative to full screen */
	                  if (tx < draw_min_x) draw_min_x = tx;
	                  if (ty < draw_min_y) draw_min_y = ty;
	                  if (tx > draw_max_x) draw_max_x = tx;
	                  if (ty > draw_max_y) draw_max_y = ty;

	                  /* draw thicker point for nicer drawing */
	                  BSP_LCD_FillRect(tx-1, ty-1, 3, 3);
	              }

	              HAL_Delay(20);
	              continue;
	          } else {
	              /* No touch now */
	              if (was_touching) {
	                  /* user just released -> perform capture + inference */

	                  /* small settle delay to avoid transient frames */
	                  HAL_Delay(150);

	                  /* Bound bbox and add padding, ensure inside screen */
	                  if (draw_max_x == 0 && draw_max_y == 0) {
	                      /* nothing drawn -> show message */
	                      Error_Display("No stroke detected");
	                  } else {
	                      int bx0 = draw_min_x - BBOX_PAD; if (bx0 < 0) bx0 = 0;
	                      int by0 = draw_min_y - BBOX_PAD; if (by0 < 0) by0 = 0;
	                      int bx1 = draw_max_x + BBOX_PAD; if (bx1 >= SRC_WIDTH) bx1 = SRC_WIDTH - 1;
	                      int by1 = draw_max_y + BBOX_PAD; if (by1 >= SRC_HEIGHT) by1 = SRC_HEIGHT - 1;

	                      /* capture and resize into mnist_img */
	                      capture_frame_bbox_to_mnist(mnist_img, bx0, by0, bx1, by1);

	                      /* Zero buffers (safety) */
	                      for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) in_data[i] = 0.0f;
	                      for (int i = 0; i < AI_NETWORK_OUT_1_SIZE; i++) out_data[i] = 0.0f;

	                      prediction = ai_run_with_conf(mnist_img);
	                      if (prediction < 0) {
	                          /* ai run error already displayed by ai_run */
	                      } else {
	                          /* Display predicted digit and confidence */
	                          /* Find top1 confidence */
	                          float top_val = out_data[0];
	                          for (int i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
	                              if (out_data[i] > top_val) top_val = out_data[i];
	                          }

	                          /* Clear result area */
	                          BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
	                          BSP_LCD_FillRect(RESULT_X+1, RESULT_Y+1, RESULT_W-2, RESULT_H-2);

	                          /* Choose color by confidence */
	                          if (top_val > 0.8f) BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
	                          else if (top_val > 0.5f) BSP_LCD_SetTextColor(LCD_COLOR_ORANGE);
	                          else BSP_LCD_SetTextColor(LCD_COLOR_RED);

	                          char resbuf[64];
	                          sprintf(resbuf, "Predicted: %d (%.1f%%)", prediction, top_val * 100.0f);
	                          BSP_LCD_DisplayStringAt(0, RESULT_Y + 12, (uint8_t*)resbuf, CENTER_MODE);

	                          /* Draw a mini histogram of the 10 probs */
	                          int bar_x = RESULT_X + 8;
	                          int bar_y = RESULT_Y + 40;
	                          int bar_w = (RESULT_W - 16) / 10;
	                          int bar_max_h = RESULT_H - 60;
	                          BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
	                          for (int i = 0; i < 10; i++) {
	                              float v = out_data[i];
	                              if (v < 0.0f) v = 0.0f;
	                              if (v > 1.0f) v = 1.0f;
	                              int h = (int)(v * bar_max_h + 0.5f);
	                              /* Draw background bar */
	                              BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
	                              BSP_LCD_FillRect(bar_x + i*bar_w, bar_y, bar_w - 2, bar_max_h);
	                              /* Draw filled bar */
	                              BSP_LCD_SetBackColor(LCD_COLOR_BLUE);
	                              BSP_LCD_FillRect(bar_x + i*bar_w, bar_y + (bar_max_h - h), bar_w - 2, h);
	                              /* Label below */
	                              char t[3]; sprintf(t, "%d", i);
	                              BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
	                              BSP_LCD_DisplayStringAt(bar_x + i*bar_w, bar_y + bar_max_h + 2, (uint8_t*)t, LEFT_MODE);
	                          }
	                      }
	                  }

	                  /* Clear drawing area so next digit starts fresh */
	                  BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
	                  BSP_LCD_FillRect(DRAW_X + 1, DRAW_Y + 1, DRAW_W - 2, DRAW_H - 2);

	                  /* Reset bbox and state */
	                  draw_min_x = SRC_WIDTH; draw_min_y = SRC_HEIGHT;
	                  draw_max_x = 0; draw_max_y = 0;
	                  was_touching = 0;
	              }
	          }

	          HAL_Delay(20);
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 6;
  RCC_OscInitStruct.PLL.PLLR = 6;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief DMA2D Initialization Function
  * @param None
  * @retval None
  */
static void MX_DMA2D_Init(void)
{

  /* USER CODE BEGIN DMA2D_Init 0 */

  /* USER CODE END DMA2D_Init 0 */

  /* USER CODE BEGIN DMA2D_Init 1 */

  /* USER CODE END DMA2D_Init 1 */
  hdma2d.Instance = DMA2D;
  hdma2d.Init.Mode = DMA2D_M2M;
  hdma2d.Init.ColorMode = DMA2D_OUTPUT_ARGB8888;
  hdma2d.Init.OutputOffset = 0;
  hdma2d.LayerCfg[1].InputOffset = 0;
  hdma2d.LayerCfg[1].InputColorMode = DMA2D_INPUT_ARGB8888;
  hdma2d.LayerCfg[1].AlphaMode = DMA2D_NO_MODIF_ALPHA;
  hdma2d.LayerCfg[1].InputAlpha = 0;
  if (HAL_DMA2D_Init(&hdma2d) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_DMA2D_ConfigLayer(&hdma2d, 1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DMA2D_Init 2 */

  /* USER CODE END DMA2D_Init 2 */

}

/**
  * @brief DSIHOST Initialization Function
  * @param None
  * @retval None
  */
static void MX_DSIHOST_DSI_Init(void)
{

  /* USER CODE BEGIN DSIHOST_Init 0 */

  /* USER CODE END DSIHOST_Init 0 */

  DSI_PLLInitTypeDef PLLInit = {0};
  DSI_HOST_TimeoutTypeDef HostTimeouts = {0};
  DSI_PHY_TimerTypeDef PhyTimings = {0};
  DSI_VidCfgTypeDef VidCfg = {0};

  /* USER CODE BEGIN DSIHOST_Init 1 */

  /* USER CODE END DSIHOST_Init 1 */
  hdsi.Instance = DSI;
  hdsi.Init.AutomaticClockLaneControl = DSI_AUTO_CLK_LANE_CTRL_DISABLE;
  hdsi.Init.TXEscapeCkdiv = 4;
  hdsi.Init.NumberOfLanes = DSI_TWO_DATA_LANES;
  PLLInit.PLLNDIV = 125;
  PLLInit.PLLIDF = DSI_PLL_IN_DIV2;
  PLLInit.PLLODF = DSI_PLL_OUT_DIV1;
  if (HAL_DSI_Init(&hdsi, &PLLInit) != HAL_OK)
  {
    Error_Handler();
  }
  HostTimeouts.TimeoutCkdiv = 1;
  HostTimeouts.HighSpeedTransmissionTimeout = 0;
  HostTimeouts.LowPowerReceptionTimeout = 0;
  HostTimeouts.HighSpeedReadTimeout = 0;
  HostTimeouts.LowPowerReadTimeout = 0;
  HostTimeouts.HighSpeedWriteTimeout = 0;
  HostTimeouts.HighSpeedWritePrespMode = DSI_HS_PM_DISABLE;
  HostTimeouts.LowPowerWriteTimeout = 0;
  HostTimeouts.BTATimeout = 0;
  if (HAL_DSI_ConfigHostTimeouts(&hdsi, &HostTimeouts) != HAL_OK)
  {
    Error_Handler();
  }
  PhyTimings.ClockLaneHS2LPTime = 28;
  PhyTimings.ClockLaneLP2HSTime = 33;
  PhyTimings.DataLaneHS2LPTime = 15;
  PhyTimings.DataLaneLP2HSTime = 25;
  PhyTimings.DataLaneMaxReadTime = 0;
  PhyTimings.StopWaitTime = 0;
  if (HAL_DSI_ConfigPhyTimer(&hdsi, &PhyTimings) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_DSI_ConfigFlowControl(&hdsi, DSI_FLOW_CONTROL_BTA) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_DSI_SetLowPowerRXFilter(&hdsi, 10000) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_DSI_ConfigErrorMonitor(&hdsi, HAL_DSI_ERROR_NONE) != HAL_OK)
  {
    Error_Handler();
  }
  VidCfg.VirtualChannelID = 0;
  VidCfg.ColorCoding = DSI_RGB888;
  VidCfg.LooselyPacked = DSI_LOOSELY_PACKED_DISABLE;
  VidCfg.Mode = DSI_VID_MODE_NB_PULSES;
  VidCfg.PacketSize = 1;
  VidCfg.NumberOfChunks = 800;
  VidCfg.NullPacketSize = 0;
  VidCfg.HSPolarity = DSI_HSYNC_ACTIVE_LOW;
  VidCfg.VSPolarity = DSI_VSYNC_ACTIVE_LOW;
  VidCfg.DEPolarity = DSI_DATA_ENABLE_ACTIVE_HIGH;
  VidCfg.HorizontalSyncActive = 14;
  VidCfg.HorizontalBackPorch = 12;
  VidCfg.HorizontalLine = 1425;
  VidCfg.VerticalSyncActive = 4;
  VidCfg.VerticalBackPorch = 2;
  VidCfg.VerticalFrontPorch = 2;
  VidCfg.VerticalActive = 480;
  VidCfg.LPCommandEnable = DSI_LP_COMMAND_DISABLE;
  VidCfg.LPLargestPacketSize = 0;
  VidCfg.LPVACTLargestPacketSize = 0;
  VidCfg.LPHorizontalFrontPorchEnable = DSI_LP_HFP_DISABLE;
  VidCfg.LPHorizontalBackPorchEnable = DSI_LP_HBP_DISABLE;
  VidCfg.LPVerticalActiveEnable = DSI_LP_VACT_DISABLE;
  VidCfg.LPVerticalFrontPorchEnable = DSI_LP_VFP_DISABLE;
  VidCfg.LPVerticalBackPorchEnable = DSI_LP_VBP_DISABLE;
  VidCfg.LPVerticalSyncActiveEnable = DSI_LP_VSYNC_DISABLE;
  VidCfg.FrameBTAAcknowledgeEnable = DSI_FBTAA_DISABLE;
  if (HAL_DSI_ConfigVideoMode(&hdsi, &VidCfg) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_DSI_SetGenericVCID(&hdsi, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DSIHOST_Init 2 */

  /* USER CODE END DSIHOST_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief LTDC Initialization Function
  * @param None
  * @retval None
  */
static void MX_LTDC_Init(void)
{

  /* USER CODE BEGIN LTDC_Init 0 */

  /* USER CODE END LTDC_Init 0 */

  LTDC_LayerCfgTypeDef pLayerCfg = {0};

  /* USER CODE BEGIN LTDC_Init 1 */

  /* USER CODE END LTDC_Init 1 */
  hltdc.Instance = LTDC;
  hltdc.Init.HSPolarity = LTDC_HSPOLARITY_AL;
  hltdc.Init.VSPolarity = LTDC_VSPOLARITY_AL;
  hltdc.Init.DEPolarity = LTDC_DEPOLARITY_AL;
  hltdc.Init.PCPolarity = LTDC_PCPOLARITY_IPC;
  hltdc.Init.HorizontalSync = 7;
  hltdc.Init.VerticalSync = 3;
  hltdc.Init.AccumulatedHBP = 14;
  hltdc.Init.AccumulatedVBP = 5;
  hltdc.Init.AccumulatedActiveW = 814;
  hltdc.Init.AccumulatedActiveH = 485;
  hltdc.Init.TotalWidth = 820;
  hltdc.Init.TotalHeigh = 487;
  hltdc.Init.Backcolor.Blue = 0;
  hltdc.Init.Backcolor.Green = 0;
  hltdc.Init.Backcolor.Red = 0;
  if (HAL_LTDC_Init(&hltdc) != HAL_OK)
  {
    Error_Handler();
  }
  pLayerCfg.WindowX0 = 0;
  pLayerCfg.WindowX1 = 800;
  pLayerCfg.WindowY0 = 0;
  pLayerCfg.WindowY1 = 480;
  pLayerCfg.PixelFormat = LTDC_PIXEL_FORMAT_ARGB8888;
  pLayerCfg.Alpha = 0;
  pLayerCfg.Alpha0 = 0;
  pLayerCfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_CA;
  pLayerCfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_CA;
  pLayerCfg.FBStartAdress = 0xC0000000;
  pLayerCfg.ImageWidth = 0;
  pLayerCfg.ImageHeight = 0;
  pLayerCfg.Backcolor.Blue = 0;
  pLayerCfg.Backcolor.Green = 0;
  pLayerCfg.Backcolor.Red = 0;
  if (HAL_LTDC_ConfigLayer(&hltdc, &pLayerCfg, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LTDC_Init 2 */

  /* USER CODE END LTDC_Init 2 */

}

/* FMC initialization function */
static void MX_FMC_Init(void)
{

  /* USER CODE BEGIN FMC_Init 0 */

  /* USER CODE END FMC_Init 0 */

  FMC_SDRAM_TimingTypeDef SdramTiming = {0};

  /* USER CODE BEGIN FMC_Init 1 */

  /* USER CODE END FMC_Init 1 */

  /** Perform the SDRAM1 memory initialization sequence
  */
  hsdram1.Instance = FMC_SDRAM_DEVICE;
  /* hsdram1.Init */
  hsdram1.Init.SDBank = FMC_SDRAM_BANK1;
  hsdram1.Init.ColumnBitsNumber = FMC_SDRAM_COLUMN_BITS_NUM_8;
  hsdram1.Init.RowBitsNumber = FMC_SDRAM_ROW_BITS_NUM_12;
  hsdram1.Init.MemoryDataWidth = FMC_SDRAM_MEM_BUS_WIDTH_32;
  hsdram1.Init.InternalBankNumber = FMC_SDRAM_INTERN_BANKS_NUM_4;
  hsdram1.Init.CASLatency = FMC_SDRAM_CAS_LATENCY_3;
  hsdram1.Init.WriteProtection = FMC_SDRAM_WRITE_PROTECTION_DISABLE;
  hsdram1.Init.SDClockPeriod = FMC_SDRAM_CLOCK_DISABLE;
  hsdram1.Init.ReadBurst = FMC_SDRAM_RBURST_DISABLE;
  hsdram1.Init.ReadPipeDelay = FMC_SDRAM_RPIPE_DELAY_2;
  /* SdramTiming */
  SdramTiming.LoadToActiveDelay = 2;
  SdramTiming.ExitSelfRefreshDelay = 7;
  SdramTiming.SelfRefreshTime = 4;
  SdramTiming.RowCycleDelay = 7;
  SdramTiming.WriteRecoveryTime = 3;
  SdramTiming.RPDelay = 2;
  SdramTiming.RCDDelay = 2;

  if (HAL_SDRAM_Init(&hsdram1, &SdramTiming) != HAL_OK)
  {
    Error_Handler( );
  }

  /* USER CODE BEGIN FMC_Init 2 */

  /* USER CODE END FMC_Init 2 */
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOI_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  BSP_LCD_SetBackColor(LCD_COLOR_RED);
      BSP_LCD_Clear(LCD_COLOR_RED);
      BSP_LCD_DisplayStringAt(0, 120, (uint8_t *)"ERROR", CENTER_MODE);
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
