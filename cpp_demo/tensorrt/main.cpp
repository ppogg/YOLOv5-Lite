#include "v5lite.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Please design config file and folder name!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string folder_name = argv[2];
    V5lite V5lite(config_file);
    V5lite.LoadEngine();
    V5lite.InferenceFolder(folder_name);
    return 0;
}