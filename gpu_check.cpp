#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <glob.h>
#include <filesystem> 

namespace fs = std::filesystem;

// --- MACROS ---
// We use ##__VA_ARGS__ to handle cases with and without extra arguments cleanly
#define GGML_LOG_INFO(fmt, ...) fprintf(stdout, "[INFO] " fmt, ##__VA_ARGS__)
#define GGML_LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt, ##__VA_ARGS__)

int ggml_hip_get_device_memory(const char *id, size_t *free, size_t *total) {
    GGML_LOG_INFO("%s searching for device %s\n", __func__, id);
    const std::string drmDeviceGlob = "/sys/class/drm/card*/device/uevent";
    const std::string drmUeventPCISlotLabel = "PCI_SLOT_NAME=";
    
    const std::string vramTotalFile = "mem_info_vram_total";
    const std::string vramUsedFile  = "mem_info_vram_used";
    const std::string gttTotalFile = "mem_info_gtt_total";
    const std::string gttUsedFile  = "mem_info_gtt_used";

    glob_t glob_result;
    glob(drmDeviceGlob.c_str(), GLOB_NOSORT, NULL, &glob_result);

    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        const char* device_file = glob_result.gl_pathv[i];
        std::ifstream file(device_file);
        if (!file.is_open()) continue;

        std::string line;
        while (std::getline(file, line)) {
            if (line.find(drmUeventPCISlotLabel) == 0) {
                std::istringstream iss(line.substr(drmUeventPCISlotLabel.size()));
                std::string pciSlot;
                iss >> pciSlot;

                if (pciSlot == std::string(id)) {
                    std::string dir = fs::path(device_file).parent_path().string();

                    // 1. READ VRAM TOTAL
                    std::ifstream vramStream((dir + "/" + vramTotalFile).c_str());
                    if (!vramStream.is_open()) {
                        file.close(); globfree(&glob_result); return 1;
                    }
                    uint64_t vram_t;
                    vramStream >> vram_t;

                    // 2. READ GTT TOTAL
                    std::ifstream gttStream((dir + "/" + gttTotalFile).c_str());
                    uint64_t gtt_t = 0;
                    if (gttStream.is_open()) {
                        gttStream >> gtt_t;
                    }

                    // 3. HEURISTIC: Check for iGPU/APU
                    // If reported VRAM is < 1GB and GTT is larger, assume Shared Memory.
                    const uint64_t APU_THRESHOLD = 1024 * 1024 * 1024; // 1 GB
                    bool use_gtt = (vram_t < APU_THRESHOLD) && (gtt_t > vram_t);

                    std::string targetTotalFile = use_gtt ? gttTotalFile : vramTotalFile;
                    std::string targetUsedFile  = use_gtt ? gttUsedFile  : vramUsedFile;

                    if (use_gtt) {
                        GGML_LOG_INFO("Detected APU/iGPU (VRAM %lu bytes < 1GB). Reporting GTT.\n", vram_t);
                        *total = gtt_t;
                    } else {
                        // FIXED: Removed the extra "" argument here to stop the warning
                        GGML_LOG_INFO("Detected Discrete/HBM GPU. Reporting VRAM.\n");
                        *total = vram_t;
                    }

                    // 4. READ USED MEMORY
                    std::ifstream usedStream((dir + "/" + targetUsedFile).c_str());
                    if (!usedStream.is_open()) {
                         file.close(); globfree(&glob_result); return 1;
                    }
                    uint64_t mem_used;
                    usedStream >> mem_used;
                    
                    *free = *total - mem_used;

                    file.close();
                    globfree(&glob_result);
                    return 0;
                }
            }
        }
        file.close();
    }
    globfree(&glob_result);
    return 1;
}

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
        std::cout << "Success reading device: " << pci_id << std::endl;
        std::cout << "Total Memory: " << total_mem / (1024*1024) << " MB" << std::endl;
        std::cout << "Free Memory:  " << free_mem / (1024*1024) << " MB" << std::endl;
        std::cout << "-----------------------------" << std::endl;
    } else {
        std::cerr << "Failed to find device with ID: " << pci_id << std::endl;
    }
    return result;
}
