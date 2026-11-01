// Report which logical CPUs share a physical core and which share a cache, so the
// SMT experiments know that, e.g., {0,1} are siblings and {0,2,4,6} are distinct cores.
#include <windows.h>

#include <cstdio>
#include <vector>

#include "../common/common.h"

static std::string mask_list(KAFFINITY m) {
    std::string s;
    for (int i = 0; i < 64; ++i)
        if (m & ((KAFFINITY)1 << i)) s += (s.empty() ? "" : ",") + std::to_string(i);
    return s;
}

int main() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    std::vector<char> buf(len);
    auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationAll, base, &len)) {
        std::fprintf(stderr, "GetLogicalProcessorInformationEx failed\n");
        return 1;
    }
    for (char* p = buf.data(); p < buf.data() + len;) {
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        Json j;
        if (info->Relationship == RelationProcessorCore) {
            j.kv("kind", "core").kv("smt", (int)(info->Processor.Flags == LTP_PC_SMT))
                .kv("cpus", mask_list(info->Processor.GroupMask[0].Mask)).emit();
        } else if (info->Relationship == RelationCache) {
            const CACHE_RELATIONSHIP& c = info->Cache;
            j.kv("kind", "cache").kv("level", (int)c.Level).kv("kib", (long long)(c.CacheSize / 1024))
                .kv("type", c.Type == CacheData ? "data" : c.Type == CacheInstruction ? "instr" : "unified")
                .kv("line", (int)c.LineSize).kv("cpus", mask_list(c.GroupMask.Mask)).emit();
        } else if (info->Relationship == RelationNumaNode) {
            j.kv("kind", "numa").kv("node", (long long)info->NumaNode.NodeNumber)
                .kv("cpus", mask_list(info->NumaNode.GroupMask.Mask)).emit();
        } else if (info->Relationship == RelationProcessorPackage) {
            j.kv("kind", "package").kv("cpus", mask_list(info->Processor.GroupMask[0].Mask)).emit();
        }
        p += info->Size;
    }
    return 0;
}
