#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <glob.h>
#include <filesystem> 

namespace fs = std::filesystem;

// --- MOCK LOGGING MACROS ---
#define GGML_LOG_INFO(fmt, ...) fprintf(stdout, "[INFO] " fmt, ##__VA_ARGS__)
#define GGML_LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt, ##__VA_ARGS__)

// --- ORIGINAL FUNCTION (No APU/GTT Logic) ---
int ggml_hip_get_device_memory(const char *id, size_t *free, size_t *total) {
    GGML_LOG_INFO("%s searching for device %s\n", __func__, id);
    const std::string drmDeviceGlob = "/sys/class/drm/card*/device/uevent";
    
    // Original file targets
    const std::string drmTotalMemoryFile = "mem_info_vram_total";
    const std::string drmUsedMemoryFile = "mem_info_vram_used";
    const std::string drmUeventPCISlotLabel = "PCI_SLOT_NAME=";

    glob_t glob_result;
    glob(drmDeviceGlob.c_str(), GLOB_NOSORT, NULL, &glob_result);

    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        const char* device_file = glob_result.gl_pathv[i];
        std::ifstream file(device_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open sysfs node" << std::endl;
            globfree(&glob_result);
            return 1;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Check for PCI_SLOT_NAME label
            if (line.find(drmUeventPCISlotLabel) == 0) {
                std::istringstream iss(line.substr(drmUeventPCISlotLabel.size()));
                std::string pciSlot;
                iss >> pciSlot;
                
                if (pciSlot == std::string(id)) {
                    std::string dir = fs::path(device_file).parent_path().string();

                    std::string totalFile = dir + "/" + drmTotalMemoryFile;
                    std::ifstream totalFileStream(totalFile.c_str());
                    if (!totalFileStream.is_open()) {
                        GGML_LOG_DEBUG("%s Failed to read sysfs node %s\n", __func__, totalFile.c_str());
                        file.close();
                        globfree(&glob_result);
                        return 1;
                    }

                    uint64_t memory;
                    totalFileStream >> memory;
                    *total = memory;

                    std::string usedFile = dir + "/" + drmUsedMemoryFile;
                    std::ifstream usedFileStream(usedFile.c_str());
                    if (!usedFileStream.is_open()) {
                        GGML_LOG_DEBUG("%s Failed to read sysfs node %s\n", __func__, usedFile.c_str());
                        file.close();
                        globfree(&glob_result);
                        return 1;
                    }

                    uint64_t memoryUsed;
                    usedFileStream >> memoryUsed;
                    *free = memory - memoryUsed;

                    file.close();
                    globfree(&glob_result);
                    return 0;
                }
            }
        }
        file.close();
    }
    GGML_LOG_DEBUG("%s unable to find matching device\n", __func__);
    globfree(&glob_result);
    return 1;
}

// --- MAIN WRAPPER ---
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <PCI_ID>" << std::endl;
        return 1;
    }

    const char* pci_id = argv[1];
    size_t free_mem = 0;
    size_t total_mem = 0;

    int result = ggml_hip_get_device_memory(pci_id, &free_mem, &total_mem);

    if (result == 0) {
        std::cout << "-----------------------------" << std::endl;
        std::cout << "ORIGINAL LOGIC RESULT:" << std::endl;
        std::cout << "Device:       " << pci_id << std::endl;
        std::cout << "Total Memory: " << total_mem / (1024*1024) << " MB" << std::endl;
        std::cout << "Free Memory:  " << free_mem / (1024*1024) << " MB" << std::endl;
        std::cout << "-----------------------------" << std::endl;
    } else {
        std::cerr << "Failed to find device." << std::endl;
    }
    return result;
}
