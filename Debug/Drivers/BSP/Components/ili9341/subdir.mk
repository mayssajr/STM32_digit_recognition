################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/BSP/Components/ili9341/ili9341.c 

OBJS += \
./Drivers/BSP/Components/ili9341/ili9341.o 

C_DEPS += \
./Drivers/BSP/Components/ili9341/ili9341.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/BSP/Components/ili9341/%.o Drivers/BSP/Components/ili9341/%.su Drivers/BSP/Components/ili9341/%.cyclo: ../Drivers/BSP/Components/ili9341/%.c Drivers/BSP/Components/ili9341/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F469xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Drivers/BSP/STM32469I-Discovery -I../Drivers/BSP/Components/otm8009a -I../Drivers/BSP/Components -I../Drivers/BSP/Components/ft6x06 -I"C:/Users/lenovo/STM32CubeIDE/workspace_1.19.0/test/Utilities/Fonts" -I"C:/Users/lenovo/STM32CubeIDE/workspace_1.19.0/test/Utilities" -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Middlewares/ST/AI/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-BSP-2f-Components-2f-ili9341

clean-Drivers-2f-BSP-2f-Components-2f-ili9341:
	-$(RM) ./Drivers/BSP/Components/ili9341/ili9341.cyclo ./Drivers/BSP/Components/ili9341/ili9341.d ./Drivers/BSP/Components/ili9341/ili9341.o ./Drivers/BSP/Components/ili9341/ili9341.su

.PHONY: clean-Drivers-2f-BSP-2f-Components-2f-ili9341

