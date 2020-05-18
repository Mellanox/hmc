/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include "hmc_mcast.h"
#include <glob.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <ifaddrs.h>

#define HMC_WHILE_IB_IF(dev_list)                                       \
    int first_time = 1;                                                 \
    char if_name[MAX_STR_LEN];                                          \
    char *saveptr = NULL;                                               \
    while(hmc_get_next_ib_if(dev_list, if_name, &first_time, &saveptr)) \

#define MAX_STR_LEN 128

static
int hmc_get_ipoib_ip(char *ifname, struct sockaddr_storage *addr) {
    struct ifaddrs *ifaddr, *ifa;
    int family, n, is_ipv4 = 0;
    char host[1025];
    const char* host_ptr;
    int rval, ret = 0, is_up;

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return ret;
    }

    for (ifa = ifaddr, n = 0; ifa != NULL; ifa=ifa->ifa_next, n++) {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6)
            continue;

        is_up = (ifa->ifa_flags & IFF_UP) == IFF_UP;
        is_ipv4 = (family == AF_INET) ? 1 : 0;

        if (is_up && !strncmp(ifa->ifa_name, ifname, strlen(ifname)) ) {
            if (is_ipv4) {
                memcpy((struct sockaddr_in *) addr,
                       (struct sockaddr_in *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in));
            }
            else {
                memcpy((struct sockaddr_in6 *) addr,
                       (struct sockaddr_in6 *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in6));
            }
#ifdef ENABLE_DEBUG
            char *env = getenv("HMC_VERBS_VERBOSE");
            int en = (env != NULL) ? atoi(env) : 0;

            if (en > 0) {
                host_ptr =
                    inet_ntop((is_ipv4) ? AF_INET : AF_INET6,
                              (is_ipv4) ? (void *)&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr :
                                          (void *)&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr,
                              host, 1024);

                if (NULL != host_ptr) {
                    printf("%-8s %s (%d) (%-3s) (addr: <%s>)\n", ifa->ifa_name,
                           (family == AF_PACKET) ? "AF_PACKET" :
                           (family == AF_INET) ? "AF_INET" :
                           (family == AF_INET6) ? "AF_INET6" : "???", family,
                           (is_up == IFF_UP) ? "UP" : "DOWN",
                           host);
                }
                else {
                   HMC_ERR("inet_ntop() failed: %d %s\n", errno, strerror(errno));
                }
            }
#endif
            ret = 1;
            break;
        }
    }
    freeifaddrs(ifaddr);
    return ret;
}

static int cmp_files(char *f1, char *f2) {
    int answer = 0;
    FILE *fp1, *fp2;

    if ((fp1 = fopen(f1, "r")) == NULL)
        goto out;
    else if ((fp2 = fopen(f2, "r")) == NULL)
        goto close;

    int ch1 = getc(fp1);
    int ch2 = getc(fp2);

    while((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2))
    {
        ch1 = getc(fp1);
        ch2 = getc(fp2) ;
    }

    if (ch1 == ch2)
        answer = 1;

    fclose(fp2);
close:
    fclose(fp1);
out:
    return answer;
}

static int port_from_file(char *port_file) {
    char buf1[MAX_STR_LEN], buf2[MAX_STR_LEN];
    FILE *fp;
    int res = -1;

    if ((fp = fopen(port_file, "r")) == NULL)
        return -1;

    if (fgets(buf1, MAX_STR_LEN - 1, fp) == NULL)
        goto out;

    int len = strlen(buf1) - 2;
    strncpy(buf2, buf1 + 2, len);
    buf2[len] = 0;
    res = atoi(buf2);

out:
    fclose(fp);
    return res;
}

#define PREF "/sys/class/net/"
#define SUFF "/device/resource"

static int dev2if(char *dev_name, char *port, char *if_name) {
    char dev_file[MAX_STR_LEN], port_file[MAX_STR_LEN], net_file[MAX_STR_LEN];
    char glob_path[MAX_STR_LEN];
    glob_t glob_el = {0,};
    int found = 0;

    char *env = getenv("HMC_IPOIB_NET_FILE_PREFIX");
    if (env != NULL) {
        sprintf(glob_path, PREF"%s*", env);
    } else {
        sprintf(glob_path, PREF"*");
    }

    sprintf(dev_file, "/sys/class/infiniband/%s"SUFF, dev_name);

    glob(glob_path, 0, 0, &glob_el);
    char **p = glob_el.gl_pathv;

    if (glob_el.gl_pathc >= 1)
        for(int i = 0; i < glob_el.gl_pathc; i++, p++)
        {
            sprintf(port_file, "%s/dev_id", *p);
            sprintf(net_file,  "%s"SUFF,    *p);
            if(cmp_files(net_file, dev_file) && port != NULL &&
               port_from_file(port_file) == atoi(port) - 1)
            {
                found = 1;
                break;
            }
        }

    globfree(&glob_el);

    if(found)
    {
        int len = strlen(net_file) - strlen(PREF) - strlen(SUFF);
        strncpy(if_name, net_file + strlen(PREF), len);
        if_name[len] = 0;
    }
    else
        strcpy(if_name, "");

    return found;
}

#define LOOK_AT_IB_LIST(_ib, _ret, _str)                    \
    do                                                      \
    {                                                       \
        if(ib_dev_list == NULL || !strcmp(ib_dev_list, "")) \
        {                                                   \
            strcpy(if_name, _ib);                           \
            return _ret;                                    \
        }                                                   \
        el = strtok_r(_str, ", ", saveptr1);                \
    }                                                       \
    while(0)

static
uintptr_t hmc_get_next_ib_if(char *ib_dev_list, char *if_name,
                               int *first_time, char **saveptr1) {
    char *dev = NULL, *port = NULL, *el;
    char dev_list[MAX_STR_LEN];

    if(*first_time) {
        *first_time = 0;
        if(ib_dev_list)
            strncpy(dev_list, ib_dev_list, MAX_STR_LEN);
        LOOK_AT_IB_LIST("ib", 1, dev_list);
    } else {
        LOOK_AT_IB_LIST("", 0, NULL);
    }

    if(el) {
        char *saveptr2 = NULL;
        dev  = strtok_r(el,   ":", &saveptr2);
        port = strtok_r(NULL, ":", &saveptr2);
        if(dev)
            dev2if(dev, port, if_name);
    }
    return (uintptr_t)dev;
}

int hmc_probe_ip_over_ib(char* ib_dev_list, struct sockaddr_storage *addr) {
    int ret = 0;
    struct sockaddr_storage rdma_src_addr;

    HMC_WHILE_IB_IF(ib_dev_list)
        if(strcmp(if_name, "") && (ret = hmc_get_ipoib_ip(if_name, &rdma_src_addr)) > 0)
            break;
    if (addr) {
        *addr = rdma_src_addr;
    }
    return ret > 0 ? HMC_SUCCESS : HMC_ERROR;
}
