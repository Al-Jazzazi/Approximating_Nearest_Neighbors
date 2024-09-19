#ifndef PAIRING_HEAP_H
#define PAIRING_HEAP_H
#include <iostream>
#include <memory>
#include <vector>

template<typename T>
class PairingHeap {
public:
    struct Node {
        T key;
        std::unique_ptr<Node> child;
        Node* sibling;

        explicit Node(T k) : key(k), child(nullptr), sibling(nullptr) {}
    };

    PairingHeap() : root(nullptr) {}

    Node* emplace(T key) {
        auto newNode = std::make_unique<Node>(key);
        Node* nodePtr = newNode.get();
        root = merge(root, newNode.release());
        return nodePtr;
    }

    T top() const {
        if (!root) throw std::runtime_error("Heap is empty");
        return root->key;
    }

    void pop() {
        if (!root) throw std::runtime_error("Heap is empty");
        Node* oldRoot = root;
        root = mergePairs(root->child.release());
        delete oldRoot;
    }

    bool empty(){
        return root == nullptr;
    }


private:
    Node* root;

    Node* merge(Node* a, Node* b) {
        if (!a) return b;
        if (!b) return a;
        if (a->key < b->key) {
            b->sibling = a->child.release();
            a->child.reset(b);
            return a;
        } else {
            a->sibling = b->child.release();
            b->child.reset(a);
            return b;
        }
    }

    Node* mergePairs(Node* node) {
        if (!node || !node->sibling) return node;
        Node* nextPair = node->sibling->sibling;
        Node* merged = merge(node, node->sibling);
        return merge(merged, mergePairs(nextPair));
    }

  
};


#endif