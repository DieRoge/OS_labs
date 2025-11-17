cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Включение директорий с заголовочными файлами
include_directories(Include)

# Поиск исходных файлов
file(GLOB SOURCES "src/*.cpp" "main.cpp")

# Создание исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES})

# Настройка свойств компиляции
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Установка свойств для разных компиляторов
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
endif()

# Опционально: настройка для многопоточности
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)