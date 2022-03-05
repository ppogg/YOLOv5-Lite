//
// Created by linghu8812 on 2021/2/8.
//

#include "common.h"
#include <cstring>

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}

std::vector<std::string>readFolder(const std::string &image_path)
{
    std::vector<std::string> image_names;
    auto dir = opendir(image_path.c_str());

    if ((dir) != nullptr)
    {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry)
        {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            image_names.push_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}

std::map<int, std::string> readImageNetLabel(const std::string &fileName)
{
    std::map<int, std::string> imagenet_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    while (getline(file, strLine))
    {
        int pos1 = strLine.find(":");
        std::string first = strLine.substr(0, pos1);
        int pos2 = strLine.find_last_of("'");
        std::string second = strLine.substr(pos1 + 3, pos2 - pos1 - 3);
        imagenet_label.insert({atoi(first.c_str()), second});
    }
    file.close();
    return imagenet_label;
}

std::map<int, std::string> readCOCOLabel(const std::string &fileName)
{
    std::map<int, std::string> coco_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    int index = 0;
    while (getline(file, strLine))
    {
        coco_label.insert({index, strLine});
        index++;
    }
    file.close();
    return coco_label;
}